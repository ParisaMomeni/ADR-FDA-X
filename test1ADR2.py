# ============================================================
# ADR Extraction Pipeline (Step-by-step, stop after each test)
# ============================================================

import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import spacy
import scispacy
from negspacy.negation import Negex
from negspacy.termsets import termset
import scispacy.umls_linking
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

tqdm.pandas()

# -----------------------------
# 1. LOAD DATA
# -----------------------------
print("📂 Loading data ...")
df = pd.read_pickle("data/processed_data.pkl")
df["Snippet"] = df["Snippet"].astype(str)
texts = df["Snippet"].tolist()

# -----------------------------
# 2. LOAD BIOBERT NER PIPELINE
# -----------------------------
print("🧠 Loading BioBERT NER model ...")
model_name = "Ishan0612/biobert-ner-disease-ncbi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
device = 0 if torch.cuda.is_available() else -1
print(f"   Device set to: {'cuda' if device == 0 else 'cpu'}")

ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
    device=device,
)

# -----------------------------
# 3. TEST 1: SIMPLE NER EXAMPLE
# -----------------------------
def test_1_simple_ner():
    print("\n🧪 TEST 1: Simple NER sanity check\n")
    test_text = "Ozempic made me nauseous and gave me stomach pain. Fatigue and headaches too!"
    print("TEXT:", test_text)
    print("\nNER OUTPUT:")
    print(ner(test_text))


# -----------------------------
# 4. LOAD SPACY + UMLS + NEGEX
# -----------------------------
print("\n🧬 Loading spaCy + SciSpaCy + NegEx ...")
nlp = spacy.load("en_core_sci_sm")

ts = termset("en")

# UMLS linker
nlp.add_pipe(
    "scispacy_linker",
    config={"resolve_abbreviations": True, "linker_name": "umls"}
)
print("🔗 scispacy_linker loaded:", "scispacy_linker" in nlp.pipe_names)
linker = nlp.get_pipe("scispacy_linker")

# NegEx for negation detection
negex = Negex(
    nlp=nlp,
    name="negex",
    neg_termset=ts.get_patterns(),
    ent_types=["DISEASE", "SYMPTOM", "CLINICAL_ATTRIBUTE", "PATHOLOGICAL_FUNCTION"],
    extension_name="negex",
    chunk_prefix=[],
)
nlp.add_pipe("negex", last=True)

# -----------------------------
# 5. TEST 2: NER ON FIRST 10 TWEETS
# -----------------------------
def test_2_ner_on_first_10():
    print("\n🧪 TEST 2: BioBERT NER on first 10 tweets\n")
    sample_texts = texts[:10]
    results = ner(sample_texts, batch_size=4)
    for i, (text, res) in enumerate(zip(sample_texts, results)):
        print(f"\n---- Text {i} ----")
        print("Text:", text)
        print("NER Output:")
        print(res)


# -----------------------------
# 6. SYNONYM MAP (MANUAL ADR CANONICAL FORMS)
# -----------------------------
synonym_map = {
    "nauseated": "nausea",
    "felt sick": "nausea",
    "queasy": "nausea",
    "sick to my stomach": "nausea",
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
    "puked": "vomiting",
    "upchucked": "vomiting",
    "stomach pain": "abdominal pain",
    "cramps": "abdominal pain",
    "belly ache": "abdominal pain",
    "diarrhea": "diarrhea",
    "runs": "diarrhea",
    "loose stools": "diarrhea",
    "heartburn": "acid reflux",
    "indigestion": "acid reflux",
    "constipated": "constipation",
    "backed up": "constipation",
    "dry mouth": "xerostomia",
    "mouth dry": "xerostomia",
    "blurred vision": "vision blurred",
    "can’t see well": "vision blurred",
}

# Our "standard" ADR canonical vocabulary = unique values of synonym_map
adr_vocab = sorted(set(synonym_map.values()))

print("\n📚 ADR vocabulary (canonical terms):")
print(adr_vocab)


# -----------------------------
# 7. UMLS UTILITIES
# -----------------------------
def normalize_term_with_umls(term: str) -> str:
    """Try to normalize a term using UMLS canonical name via SciSpaCy linker."""
    doc = nlp(term)
    for ent in doc.ents:
        if ent._.has("umls_ents"):
            for cui, _ in ent._.umls_ents:
                try:
                    concept = linker.umls.cui_to_entity[cui]
                    return concept.canonical_name.lower()
                except KeyError:
                    continue
    # No UMLS match
    print(f"❌ No UMLS match for: {term}")
    return term.lower()


def is_umls_symptom(term: str) -> bool:
    """
    Check if UMLS says this term is a symptom / sign / finding / disease.
    Based on semantic types: T184 (sign/symptom), T033, T047, T046, etc.
    """
    doc = nlp(term)
    for ent in doc.ents:
        if ent._.has("umls_ents"):
            for cui, _ in ent._.umls_ents:
                try:
                    concept = linker.umls.cui_to_entity[cui]
                    # concept.types is a list of semantic type codes, e.g. ["T184"]
                    if any(sty in {"T184", "T033", "T047", "T046"} for sty in concept.types):
                        return True
                except KeyError:
                    continue
    return False


# -----------------------------
# 8. EMBEDDING MODEL + SEMANTIC MAPPING
# -----------------------------
print("\n🧊 Loading sentence embedding model for semantic ADR mapping ...")
#emb_model = SentenceTransformer("pritamdeka/BioBERT-mnli-snli-scinli")
emb_model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

adr_vocab_emb = emb_model.encode(
    adr_vocab, convert_to_tensor=True, show_progress_bar=False
)

def semantic_map_to_adr(term: str, threshold: float = 0.40):
    """
    Attempt to map a free-text term to one of the known ADR concepts
    using sentence embeddings + cosine similarity.
    """
    if not term.strip():
        return None

    term_emb = emb_model.encode(term, convert_to_tensor=True, show_progress_bar=False)
    # Cosine similarity between this term and all ADR concepts
    scores = util.cos_sim(term_emb, adr_vocab_emb)[0]  # shape: [num_adrs]

    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()
    best_adr = adr_vocab[best_idx]

    # Debug print (optional)
    print(f"   Semantic match: '{term}' -> '{best_adr}' (score={best_score:.3f})")

    if best_score >= threshold:
        return best_adr
    return None


# -----------------------------
# 9. NORMALIZATION FUNCTION (SYNONYM + UMLS + SEMANTIC)
# -----------------------------
def normalize_term(term: str) -> str:
    """
    Normalize a raw ADR-like term to a canonical form:
    1) manual slang mapping (synonym_map)
    2) UMLS canonical concept IF it's a symptom
    3) semantic mapping to closest known ADR
    4) if UMLS says it's a symptom but unknown: keep as new ADR
    5) else: return raw lowercased term
    """
    term_lower = term.lower().strip()

    # 1) Direct slang mapping
    if term_lower in synonym_map:
        return synonym_map[term_lower]

    # 2) UMLS canonical concept
    umls_norm = normalize_term_with_umls(term_lower)
    if umls_norm != term_lower:
        # Only trust UMLS normalization if it's a symptom-type concept
        if is_umls_symptom(umls_norm):
            return umls_norm

    # 3) Semantic mapping into known ADR vocab
    sem_norm = semantic_map_to_adr(term_lower)
    if sem_norm is not None:
        return sem_norm

    # 4) NEW: if UMLS says this (un-normalized) term is a symptom, keep it as a "new ADR"
    if is_umls_symptom(term_lower):
        return term_lower

    # 5) Fallback: raw
    return term_lower


# -----------------------------
# 10. TEST 3: NORMALIZATION OF HAND-PICKED TERMS
# -----------------------------
def test_3_normalization():
    print("\n🧪 TEST 3: UMLS + synonym + semantic normalization\n")
    test_terms = [
    "brain fog",
    "tingling fingers",
    "heart skipping",
    "jaw pain",
    "cold sweats",
    "weird dreams",
    "visual snow",
    "heavy legs",
    "burning tongue",
    "metallic taste",
    "eye twitch",
    "restless legs",
    "shaking hands",
    "sun sensitivity",
    "floaters in vision",
    "tight chest"
]

    for term in test_terms:
        print(f"\nTERM: {term}")
        norm = normalize_term(term)
        print(f" → Normalized: {norm}")


# -----------------------------
# 11. TEST 4: PURE SEMANTIC ADR MAPPING (UNKNOWN TERMS)
# -----------------------------
def test_4_semantic_mapping():
    print("\n🧪 TEST 4: Semantic ADR mapping (terms NOT in synonym_map)\n")
    semantic_test_terms = [
        "tummy ache",
        "my head is pounding",
        "woozy",
        "can't focus",
        "my face is red",
        "blur",
        "upset belly",
        "acid reflux"
    ]
    for term in semantic_test_terms:
        mapped = semantic_map_to_adr(term)
        print(f"TERM: '{term}' → semantic map: {mapped}")


# -----------------------------
# 15. TEST 6: ADR Extraction on First 50 Samples
# -----------------------------
def test_4_adr_on_first_50():
    print("\n🧪 TEST 6: Extract ADRs from first 50 text snippets\n")
    sample_texts = texts[:50]

    for i, text in enumerate(sample_texts):
        print(f"\n--- Sample {i+1} ---")
        print("Text:", text)
        adrs = extract_adrs(text)
        print("Extracted ADRs:", adrs)


# -----------------------------
# 12. ADR EXTRACTION FUNCTIONS
def is_valid_candidate(term):
    if not term:
        return False
    if len(term.split()) > 5:
        return False
    if any(x in term.lower() for x in ["http", "www", "@", "#", "https"]):
        return False
    if len(term) <= 2:
        return False
    return True

# -----------------------------

def extract_adrs(text: str):
    """
    Extract *positive* ADR mentions (no negation filtering here yet).
    Uses BioBERT NER + normalization and keeps only known ADRs
    (currently those in synonym_map.values()).
    """
    try:
        allowed_adrs = set(synonym_map.values())
        entities = ner(text[:512])
        doc = nlp(text)

        result = []
        for ent in entities:
            term = ent["word"].replace("##", "")
            if not is_valid_candidate(term):
                print(f"⚠️ Skipping non-ADR-like entity: {term}")
                continue

        print(f"📌 Raw ADR term candidate: {term}")
        normalized = normalize_term(term)
        print(f"   → Normalized: {normalized}")

        if normalized in allowed_adrs:
            print(f"✅ Valid ADR kept: {normalized}")
            result.append(normalized)


        return list(set(result))
    except Exception as e:
        print(f"Error processing text: {text[:50]} - {e}")
        return []


def extract_adrs_with_negation(text: str):
    """
    Extract ADRs and separate them into:
    - positive (not negated)
    - negated (e.g., 'no nausea', 'denies headache')
    """
    try:
        entities = ner(text[:512])
        doc = nlp(text)
        allowed_adrs = set(synonym_map.values())

        pos_adrs = set()
        neg_adrs = set()

        for ent in entities:
            term = ent["word"].replace("##", "")

            if not is_valid_candidate(term):
                continue

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
            "negated": list(neg_adrs),
        }
    except Exception as e:
        print(f"Error processing text: {text[:50]} - {e}")
        return {"positive": [], "negated": []}


# -----------------------------
# 13. FULL PIPELINE: NEGATION STATS + CSV
# -----------------------------
def run_full_pipeline_and_save():
    print("\n📊 Running negation analysis over all snippets ...\n")
    negation_stats = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Negation"):
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

    # Now build Extracted_ADRs column
    print("\n🧪 Extracting ADRs for each tweet ...\n")
    extracted_adrs = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting ADRs"):
        text = row["Snippet"]
        adrs = extract_adrs(text)
        extracted_adrs.append(adrs)
        print(f"Row {i}: {adrs}")

    df["Extracted_ADRs"] = extracted_adrs

    out_path = "data/1adrs2test.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ ADR extraction complete and results saved to: {out_path}")

# FULL PIPELINE: NEGATION STATS + CSV for 50
def test_7_full_pipeline_first_50():
    print("\n📊 Running full pipeline (with negation) on first 50 rows ...\n")
    sample_df = df.head(50)

    negation_stats = []
    extracted_adrs = []

    for i, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing"):
        text = row["Snippet"]
        adrs_info = extract_adrs_with_negation(text)
        negation_stats.append(adrs_info)

        adrs = extract_adrs(text)
        extracted_adrs.append(adrs)

        print(f"\n--- Row {i} ---")
        print("Text:", text)
        print("Negation Aware ADRs:", adrs_info)
        print("Extracted ADRs (raw):", adrs)

    sample_df["Negation_Info"] = negation_stats
    sample_df["Extracted_ADRs"] = extracted_adrs

    sample_df.to_csv("data/adr_test_first_50.csv", index=False)
    print("\n✅ Results saved to data/adr_test_first_50.csv")


# -----------------------------
# 14. MAIN: SELECT WHICH STEP TO RUN
# -----------------------------
if __name__ == "__main__":
    # 👇 CHANGE THIS VALUE TO CHOOSE WHAT RUNS
    # 1 = simple NER test
    # 2 = NER on first 10 tweets
    # 3 = normalization tests
    # 4 = semantic mapping tests
    # 5 = full pipeline (negation + extraction + CSV)
    CURRENT_STEP = 7

    if CURRENT_STEP == 1:
        test_1_simple_ner()
        sys.exit()

    if CURRENT_STEP == 2:
        test_2_ner_on_first_10()
        sys.exit()

    if CURRENT_STEP == 3:
        test_3_normalization()
        sys.exit()

    if CURRENT_STEP == 4:
        test_4_semantic_mapping()
        sys.exit
    
    if CURRENT_STEP == 6:
        test_4_adr_on_first_50()
        sys.exit()

    if CURRENT_STEP == 5:
        run_full_pipeline_and_save()
        sys.exit()

    if CURRENT_STEP == 7:
        test_7_full_pipeline_first_50()
        sys.exit()

    print("⚠️ CURRENT_STEP is not in {1,2,3,4,5}. Please set it appropriately.")
