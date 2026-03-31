import re
import emoji
from spellchecker import SpellChecker
import spacy
import pandas as pd
from spacy.matcher import PhraseMatcher
from negspacy.negation import Negex
from negspacy.termsets import termset


# --- Load Models ---
nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md")  # For disease/chemical detection
nlp_clean = spacy.load("en_core_web_sm")     # For sentence segmentation, etc.

# Optional: Med7 or other ADR-specific model
try:
    nlp_adr = spacy.load("en_core_med7_lg")
except:
    nlp_adr = None  # Skip if not available

# Negex setup
ts = termset("en")
negex = Negex(
    nlp=nlp_bc5cdr,
    name="negex",
    chunk_prefix=[],
    ent_types=["DISEASE"],
    extension_name="negex",
    neg_termset=ts.get_patterns()
)
nlp_bc5cdr.add_pipe("negex", last=True)

# --- Data ---
adr_adjectives = {
   
    "unsteady", "woozy", "nervous", "dumb", "paralyzed", "frazzled", "numbed",
    "vibrating", "burned", "itchier", "raw", "twitchy", "loose", "tight", "sensitive",
    "hollow", "bubbly", "frantic", "cold", "hot", "nervy", "snappy", "dull", "blank",
    "fogged", "dizzying", "bitter", "groggy", "sore", "itchiest", "trapped", "sunken",
    "unbearable", "exhausted", "unfocused", "foggy-brained", "sniffly", "winded", "weepy",
    "on-edge", "cramped", "scratchy", "wobbly", "painful", "muzzy", "spacy", "blurry", "ticked"
}

adr_terms = {
    "muscle pain", "jaw clenching", "eye twitching", "cold sweats", "rash on arms", 
    "irregular heartbeat", "heartburn", "acid stomach", "tight chest", "dry eyes", "watery eyes",
    "heavy limbs", "drooling", "loss of appetite", "increased appetite", "insomnia", "night sweats",
    "difficulty concentrating", "speech slurring", "loss of balance", "mouth ulcers", "fever",
    "cold feet", "dizziness", "dehydration", "stiff neck", "sensitivity to sound", 
    "tingling in fingers", "bruising", "cramping", "tenderness", "nail discoloration",
    "bleeding gums", "slow heartbeat", "hot flashes", "cold flash", "ear ringing", "jaw pain",
    "redness", "vomiting blood", "black stool", "vision loss", "skin cracking", "eczema", 
    "urinary burning", "fluid retention", "fatigue", "muscle stiffness", "shivering"
}

abbreviation_map = {
    "np": "no problem",  # context-aware
    "afaik": "as far as i know",
    "asap": "as soon as possible",
    "w/o": "without",
    "w/": "with",
    "h/o": "history of",
    "b/c": "because",
    "dx": "diagnosis",
    "sx": "symptom",
    "tx": "treatment",
    "fx": "fracture",
    "cx": "complication",
    "hx": "history",
    "s/p": "status post",
    "yo": "years old",
    "otc": "over the counter",
    "prn": "as needed",
    "qhs": "every night at bedtime",
    "qd": "every day",
    "bid": "twice daily",
    "tid": "three times daily",
    "q4h": "every 4 hours",
    "pt": "patient",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "npo": "nothing by mouth",
    "iv": "intravenous",
    "po": "by mouth",
    "im": "intramuscular",
    "stat": "immediately",
    "wtf": "",
    "lmao": "",
    "ikr": "i know right",
    "ily": "i love you",
    "bruh": "",
    "ugh": "",
    "rip": "",  # avoid triggering false positives
    "lmfao": "",
    "vom": "vomit"

}

# Phrase matcher for ADR terms
matcher = PhraseMatcher(nlp_clean.vocab)
adr_patterns = [nlp_clean.make_doc(term) for term in adr_terms]
matcher.add("ADR_TERMS", adr_patterns)


# --- Utilities ---

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    for abbr, full in abbreviation_map.items():
        text = text.replace(abbr, full)

    spell = SpellChecker()
    words = text.split()
    misspelled = spell.unknown(words)
    corrected = [spell.correction(w) if w in misspelled else w for w in words]
    return " ".join(corrected)

def segment_sentences(text):
    doc = nlp_clean(text)
    return [sent.text.strip() for sent in doc.sents]

def determine_perspective(token):
    for ancestor in token.ancestors:
        if ancestor.lemma_ in {"feel", "felt", "was", "were", "be", "seem", "seemed"}:
            if any(child.text.lower() in {"i", "me", "my"} and child.dep_ == "nsubj" for child in ancestor.children):
                return "personal"
            elif any(child.text.lower() in {"others", "they", "he", "she"} and child.dep_ == "nsubj" for child in ancestor.children):
                return "impersonal"
    return "impersonal"


# --- Main ADR Detection ---

def detect_adrs(sent):
    results = {}
    doc_bc5cdr = nlp_bc5cdr(sent)
    doc_clean = nlp_clean(sent)

    # Disease mentions (from bc5cdr)
    for ent in doc_bc5cdr.ents:
        if ent.label_ == "DISEASE":
            status = "Negated" if ent._.negex else "Affirmed"
            perspective = determine_perspective(ent.root)
            results[ent.text.lower()] = (status, perspective)

    # Adjective-based symptoms
    for token in doc_bc5cdr:
        if token.pos_ == "ADJ" and token.lemma_ in adr_adjectives:
            status = "Negated" if token._.negex else "Affirmed"
            perspective = determine_perspective(token)
            results[token.text.lower()] = (status, perspective)

    # Phrase matcher (adr_terms)
    matches = matcher(doc_clean)
    for _, start, end in matches:
        span = doc_clean[start:end]
        results[span.text.lower()] = ("Affirmed", determine_perspective(span.root))

    # Optional: Use ADR-specific NER model (like Med7)
    if nlp_adr:
        doc_adr = nlp_adr(sent)
        for ent in doc_adr.ents:
            if ent.label_.lower() == "adr":
                results[ent.text.lower()] = ("Affirmed", determine_perspective(ent.root))

    return results


def process_post(post):
    normalized = normalize_text(post)
    sentences = segment_sentences(normalized)
    combined_results = {}
    for sent in sentences:
        adrs = detect_adrs(sent)
        combined_results.update(adrs)
    return combined_results


def extract_adrs(text):
    try:
        result = process_post(text)
        print(f"✅ Valid ADR found: {result}")
        return result
    except Exception as e:
        print(f"Error processing text: {text[:100]}... \nError: {e}")
        return {}


# --- Execution ---
if __name__ == "__main__":
    df = pd.read_pickle("data/processed_data.pkl").head(100)
    df["Extracted_ADRs"] = df["Snippet"].astype(str).apply(extract_adrs)
    df.to_pickle("data/adrs.pkl")
    df.to_csv("data/adrs.csv", index=False)
    print("Done! ADRs extracted and saved.")
