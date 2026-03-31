from curses.ascii import EOT
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import scispacy
from negspacy.negation import Negex
from negspacy.termsets import termset
from tqdm import tqdm
tqdm.pandas()
import scispacy.umls_linking
from spacy.tokens import Doc
from pyvis.network import Network
import webbrowser
from sentence_transformers import SentenceTransformer, util
import torch
import sys


df = pd.read_pickle("data/processed_data.pkl")
before_retweets = len(df)

df = df[df['Engagement Type'] != 'RETWEET'] #
after_retweets = len(df)

# Print results
print(f"Rows before removing retweets: {before_retweets}")
print(f"Rows after removing retweets: {after_retweets}")
print(f"Number of retweets removed: {before_retweets - after_retweets}")


# 1) Load the BioBERT model for NER
model_name = "Ishan0612/biobert-ner-disease-ncbi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1

# Setup the pipeline
ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device,  # Use GPU if available, otherwise CPU
    #truncation=True,
    #max_length=512
)
texts = df["Snippet"].tolist()



#all_ner_results = ner(texts, batch_size=16)
# Load SpaCy model and integrate negation detection
nlp = spacy.load("en_core_sci_sm")
ts = termset("en")

nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}) # Add UMLS Entity Linking via scispacy
print("🔗 scispacy_linker loaded:", "scispacy_linker" in nlp.pipe_names)
linker = nlp.get_pipe("scispacy_linker")

negex = Negex(
  nlp=nlp, 
  name="negex",
  neg_termset=ts.get_patterns(),
  ent_types=["DISEASE", "SYMPTOM", "CLINICAL_ATTRIBUTE", "PATHOLOGICAL_FUNCTION"],  # Set based on ADR labels
  extension_name="negex",
  chunk_prefix=[]  # Ensures that negation detection does not rely on specific chunk prefixes, simplifying the negation logic
)
nlp.add_pipe("negex", last=True) 

#nlp.add_pipe("negex", config={"ent_types": ["DISEASE"]})

# Domain-specific synonym mapping to normalize informal ADR phrases to clinical terms
# Inspired by UMLS and MedDRA mappings

synonym_map = {
    "nauseated": "nausea",
    #"felt sick": "nausea",
    "queasy": "nausea",
    "sick to my stomach": "nausea",
     "nauseous": "nausea",
    "feel sick to my stomach": "nausea",
    "light headed": "dizziness",
    "lightheaded": "dizziness",
    "dizzy": "dizziness",
    "woozy": "dizziness",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "wiped out": "fatigue",
    "low energy": "fatigue",
    "head hurting": "headache",
    "migraine": "headache",
    "head pain": "headache",
    "throwing up": "vomiting",
    "vomited": "vomiting",
    #"puked": "vomiting",
    "upchucked": "vomiting",
    "puked": "vomiting",
    "threw up": "vomiting",
    "stomach pain": "abdominal pain",
    "cramps": "abdominal pain",
    "belly ache": "abdominal pain",
    "diarrhea": "diarrhea",
    "runs": "diarrhea",
    "loose stools": "diarrhea",
    "loose stool": "diarrhea",
    "runny stools": "diarrhea",
    "frequent stools": "diarrhea",
    "messing up my gut": "diarrhea",
    "bloated": "bloating",
    "heartburn": "acid reflux",
    "indigestion": "acid reflux",
    "constipated": "constipation",
    "backed up": "constipation",
    "can't go": "constipation",
    "backed up": "constipation",
    "dry mouth": "xerostomia",
    "mouth dry": "xerostomia",
    "blurred vision": "vision blurred",
    "can’t see well": "vision blurred"
}

def normalize_term_with_umls(term, doc):  
    for ent in doc.ents:
        #if ent._.has("umls_ents"):
        if ent._.has("umls_ents") and ent._.umls_ents:
            for umls_ent in ent._.umls_ents:
                try:
                    concept = linker.umls.cui_to_entity[umls_ent[0]]
                    return concept.canonical_name.lower()
                except KeyError:
                    continue
    print(f"❌ No UMLS match for: {term}")
    return term.lower()

def is_umls_symptom(term, doc):
    for ent in doc.ents:
        if ent._.has("umls_ents") and ent._.umls_ents:
            for cui, _ in ent._.umls_ents:
                try:
                    concept = linker.umls.cui_to_entity[cui]
                    if any(sty in {"T184", "T033", "T047", "T046", "T191", "T020"} for sty in concept.types):
                        return True
                except KeyError:
                    continue
    return False

learned_adrs = set()
# Normalize ADR term to its canonical form
'''def normalize_term(term):
    term_lower = term.lower().strip()

    # 1) Direct slang mapping
    if term_lower in synonym_map:
        return synonym_map[term_lower]

    # 2) UMLS canonical concept
    umls_norm = normalize_term_with_umls(term_lower)  # this already returns lower()
    # if UMLS changed the term to something meaningful, use that
    if umls_norm != term_lower and is_umls_symptom(umls_norm):
        return umls_norm
#new
    # 3) Embedding-based semantic mapping to nearest ADR concept
    sem_norm = semantic_map_to_adr(term_lower)
    if sem_norm is not None:
        return sem_norm
#new   
       # 4) If UMLS says the original term is a symptom, accept it
   

     # 4) NEW FEATURE: If UMLS says a previously unknown term is a symptom
    if umls_norm == term_lower and is_umls_symptom(term_lower):
        learned_adrs.add(term_lower)
        return term_lower
    # 4) Fallback: return the raw term
    return term_lower
    '''
def normalize_term(term, doc):
    term_lower = term.lower().strip()

    if term_lower in synonym_map:
        return synonym_map[term_lower], "synonym"
    
    
    umls_norm = normalize_term_with_umls(term_lower, doc)
    if umls_norm != term_lower and is_umls_symptom(umls_norm, doc):
        return umls_norm, "umls"
    
    if umls_norm == term_lower and is_umls_symptom(term_lower, doc):
        learned_adrs.add(term_lower)
        return term_lower, "learned"

    sem_norm = semantic_map_to_adr(term_lower)
    if sem_norm is not None:
        return sem_norm, "semantic"

    
    return term_lower, "raw"





#Store UMLS CUI and concept name for later analysis # use for debug:
def get_umls_concepts(term,doc):
    #doc = nlp(term)
    concepts = []
    for ent in doc.ents:
        #if ent._.has("umls_ents"):
        if ent._.has("umls_ents") and ent._.umls_ents:
            for umls_ent in ent._.umls_ents:
                try:
                    cui = umls_ent[0]
                    name = linker.umls.cui_to_entity[cui].canonical_name
                    concepts.append((cui, name))
                except KeyError:
                    continue
    return concepts


# Unique canonical ADR terms
all_terms = set()
for text in texts:  
    doc = nlp(text)
    for ent in doc.ents:
        #if ent._.has("umls_ents"):
        if ent._.has("umls_ents") and ent._.umls_ents:
            for cui, _ in ent._.umls_ents:
                try:
                    concept = linker.umls.cui_to_entity[cui]
                    if any(sty in {"T184", "T047", "T046", "T033"} for sty in concept.types):
                        all_terms.add(concept.canonical_name.lower())
                except KeyError:
                    continue
#“What ADRs exist in my universe?”
adr_vocab = sorted(
    set(all_terms)
    .union(synonym_map.values())
    .union(learned_adrs)
)

# Load a biomedical sentence embedding model ( can pick another Sci/Bio model)
#emb_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli") #private or deleted
#emb_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-sentence")
emb_model = SentenceTransformer("all-MiniLM-L6-v2")


# Precompute embeddings (Tensor shape: [num_adrs, hidden_dim])
adr_vocab_emb = emb_model.encode(adr_vocab, convert_to_tensor=True, show_progress_bar=False)

def semantic_map_to_adr(term, threshold=0.5): # higher threshold: stricter matching (only strong semantic matches are accepted). lower threashold: ooser matching (weaker matches can be accepted more easily)
    """
    Attempt to map a free-text term to one of the known ADR concepts
    using sentence embeddings + cosine similarity.
    """
    if not term.strip():
        return None
    #Convert the term to an embedding vector
    term_emb = emb_model.encode(term, convert_to_tensor=True, show_progress_bar=False)
    # Cosine similarity between this term and all ADR concepts
    scores = util.cos_sim(term_emb, adr_vocab_emb)[0]  # shape: [num_adrs]
    
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    best_adr = adr_vocab[best_idx]
    
    # Debug:
    # print(f"Semantic match: '{term}' -> '{best_adr}' (score={best_score:.3f})")
    
    if best_score >= threshold:
        return best_adr
    return None


# -----------------------------------------------------------
# TEST 4: Semantic ADR mapping (terms NOT in synonym_map)
# -----------------------------------------------------------


# Extract ADRs from a post
'''def extract_adrs(text):
    try:
        #allowed_adrs = set(synonym_map.values())
        allowed_adrs = set(synonym_map.values()).union(learned_adrs)
        entities = ner(text[:512])
        doc = nlp(text)
        adr_labels = {'DISEASE', 'SYMPTOM', 'PATHOLOGICAL_FUNCTION', 'CLINICAL_ATTRIBUTE'}

        result = []
        for ent in entities:
            term = ent["word"].replace("##", "")
            print(f"📌 Raw ADR term: {term}")  # 👈 Before normalization

            normalized = normalize_term(term)
            print(f"🔍 Normalized (after UMLS + synonym): {normalized}")  # 👈 After UMLS/synonym

            #is_negated = any(normalized in e.text.lower() and e._.negex for e in doc.ents)
            # Negation filtering
            #if (
             #   normalized in allowed_adrs and 
             #   not any(e.text.lower() == term.lower() and e._.negex for e in doc.ents)
            #):
              #  result.append(normalized)

            if normalized in allowed_adrs:
                print(f"✅ Valid ADR found: {normalized}")  # 👈 Valid ADR
                result.append(normalized)
        return list(set(result))
    except Exception as e:
        print(f"Error processing text: {text[:50]} - {e}")
        return []
 #--------------------------------------------------------------  
 # negation detection is very likely one of the main reasons why many of the Extracted_ADRs are coming out as null (empty list [])
 # This function extracts ADRs from text while considering negation 

def extract_adrs_with_negation(text):
    try:
        entities = ner(text[:512])
        doc = nlp(text)
        allowed_adrs = set(synonym_map.values())

        pos_adrs = set()
        neg_adrs = set()

        for ent in entities:
            term = ent["word"].replace("##", "")
            normalized = normalize_term(term)

            for spacy_ent in doc.ents:
                if normalized in spacy_ent.text.lower():
                    if spacy_ent._.negex:
                        neg_adrs.add(normalized)
                    else:
                        if normalized in allowed_adrs:
                            pos_adrs.add(normalized)
        return {
            "positive": list(pos_adrs),
            "negated": list(neg_adrs)
        }
    except Exception as e:
        print(f"Error processing text: {text[:50]} - {e}")
        return {"positive": [], "negated": []}'''

'''def is_valid_adr(term):
    #if len(term.split()) > 5:
       # return False

    #if term in {"diabetes", "hypertension", "hbp", "obesity"}:
      #  return False

    if term in synonym_map.values():
        return True

    if is_umls_symptom(term):
        return True

    return False
'''
#“Should this normalized term be considered a real ADR?”
def is_valid_adr(normalized, source):

    DISEASE_TERMS = {
        "diabetes", "hypertension", "hbp", "obesity", "type 2 diabetes"
    }

    # Reject core diseases immediately
    if normalized in DISEASE_TERMS:
        return False

    # Trusted sources
    if source in {"synonym", "umls", "learned"}:
        return True

    # Semantic = cautious
    if source == "semantic":
        GENERIC_WORDS = {
            "drug", "medicine", "people", "thing", "world", "problem",
            "health", "life", "year", "day", "celebrity"
        }
        return normalized not in GENERIC_WORDS

    return False

def extract_adrs_unified(text, include_negation=False):

    try:
        doc = nlp(text)
        #allowed_adrs = set(synonym_map.values()).union(learned_adrs)
        entities = ner(text[:512])
        

        pos_adrs = set()
        neg_adrs = set()
        result = set()

        for ent in entities:
            term = ent["word"].replace("##", "").strip()
            # ✅ FILTER BAD TOKENS
            if len(term) < 3:
                continue
            if term.startswith("http") or term.startswith("@") or term.startswith("#"):
                continue
            if not any(c.isalpha() for c in term):
                continue
            print(f"📌 Raw ADR term: {term}")
            normalized, source = normalize_term(term, doc)
            print(f"🔎 {normalized}  ← source: {source}")

            if not is_valid_adr(normalized, source):
                continue
            #debug mode
            #umls_concepts = get_umls_concepts(normalized, doc)
            #print(f"🧬 UMLS Concepts for {normalized}: {umls_concepts}")
            ##normalized = normalize_term(term)
            #print(f"🔍 Normalized: {normalized}")

            #if normalized not in allowed_adrs:
                #ontinue

            #if not is_valid_adr(normalized):
               # continue

            # ✅ DEFAULT: consider POSITIVE
            is_negated = False

            # Only check negation if requested
            if include_negation:
                for spacy_ent in doc.ents:
                    if normalized in spacy_ent.text.lower() and spacy_ent._.negex:
                        is_negated = True
                        break

            if include_negation:
                if is_negated:
                    neg_adrs.add(normalized)
                else:
                    pos_adrs.add(normalized)
            else:
                result.add(normalized)

        if include_negation:
            return {
                "positive": list(pos_adrs),
                "negated": list(neg_adrs)
            }
        else:
            return list(result)

    except Exception as e:
        print(f"❌ Error processing text: {text[:50]} - {e}")
        return {"positive": [], "negated": []} if include_negation else []
negation_stats = []

'''for i, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Negation"):
    text = row["Snippet"]
    adrs_info = extract_adrs_with_negation(text)
    negation_stats.append(adrs_info)
total_pos = sum(len(d["positive"]) for d in negation_stats)
total_neg = sum(len(d["negated"]) for d in negation_stats)
total_all = total_pos + total_neg

if total_all > 0:
    negation_percent = (total_neg / total_all) * 100
    print(f"🔍 Total ADR mentions: {total_all}")
    print(f"✔️ Non-negated ADRs: {total_pos}")
    print(f"❌ Negated ADRs: {total_neg}")
    print(f"📉 Percentage Negated: {negation_percent:.2f}%")
else:
    print("⚠️ No ADR mentions found.")
#--------------------------------------------------------------

df["Snippet"] = df["Snippet"].astype(str)
#df["Extracted_ADRs"] = df["Snippet"].astype(str).apply(extract_adrs) #Duplicates


extracted_adrs = []
for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ADRs"):
    text = row["Snippet"]
    adrs = extract_adrs(text)
    extracted_adrs.append(adrs)
    print(f"Row {i}: {adrs}")'''

# Ensure text column is string
df["Snippet"] = df["Snippet"].astype(str)

# ✅ Step 1: Extract ADRs with negation info using unified function
print("🔍 Extracting ADRs with negation handling...\n")
df["ADR_Negation_Info"] = df["Snippet"].progress_apply(lambda x: extract_adrs_unified(x, include_negation=True))

# ✅ Step 2: Calculate negation stats
negation_stats = df["ADR_Negation_Info"].tolist()
total_pos = sum(len(d["positive"]) for d in negation_stats)
total_neg = sum(len(d["negated"]) for d in negation_stats)
total_all = total_pos + total_neg

if total_all > 0:
    negation_percent = (total_neg / total_all) * 100
    print(f"🔍 Total ADR mentions: {total_all}")
    print(f"✔️ Non-negated ADRs: {total_pos}")
    print(f"❌ Negated ADRs: {total_neg}")
    print(f"📉 Percentage Negated: {negation_percent:.2f}%")
else:
    print("⚠️ No ADR mentions found.")

# ✅ Step 3: Extract just the positive ADRs into a flat list column
df["Extracted_ADRs"] = df["ADR_Negation_Info"].apply(lambda d: d.get("positive", []))

# ✅ Optional: show progress
for i, adrs in enumerate(df["Extracted_ADRs"]):
    print(f"Row {i}: {adrs}")


#df["Extracted_ADRs"] = extracted_adrs

df.to_pickle("data/1adrs2.pkl")
df.to_csv("data/1adrs2.csv", index=False)

print("✅ ADR extraction complete and results saved.")
# ================================
# PIPELINE VALIDATION TEST (UNIT TEST)
# ================================

print("\n===== PIPELINE VALIDATION TEST =====")

test_sentences = [
    "I get so tired, just want to sleep after using",
    "After the injection I feel extremely sleepy and exhausted",
    "This medication makes me want to lie down all day",
    "I feel drained and have no energy since starting this",
    "Been having loose stools all day,"
    " this drug is messing up my gut."
    "I literally puked twice last night after taking the injection."
]

for text in test_sentences:
    print("\n----------------------------------")
    print("INPUT TEXT:", text)
    #extracted = extract_adrs(text)
    extracted = extract_adrs_unified(text, include_negation=True)
    print("✅ ADRs (with negation):", extracted)


# ================================
# END VALIDATION TEST
# ================================

# ------------------------



