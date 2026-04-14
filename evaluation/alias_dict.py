from __future__ import annotations

# Canonical surface form -> aliases that commonly appear in CTD / biomedical text.
# Keep this small and high-precision; the KB already contributes synonyms.
ALIAS_MAP = {
    "famotidine": ["famotin"],
    "indomethacin": ["indomethacin sodium"],
    "acetaminophen": ["paracetamol"],
    "paracetamol": ["acetaminophen"],
    "aspirin": ["acetylsalicylic acid"],
    "acetylsalicylic acid": ["aspirin"],
}
