"""Curated word lists for demonstrating gender bias in embeddings."""

PRESETS: dict[str, list[str]] = {
    "Jobs": [
        "surgeon", "doctor", "nurse", "teacher", "engineer", "programmer",
        "secretary", "soldier", "pilot", "ceo", "receptionist", "mechanic",
        "librarian", "chef", "dentist", "lawyer", "scientist", "astronaut",
        "firefighter", "babysitter", "architect", "cashier", "professor",
    ],
    "Family": [
        "mother", "father", "sister", "brother", "wife", "husband",
        "grandmother", "grandfather", "daughter", "son", "aunt", "uncle",
        "niece", "nephew", "cousin", "mama", "papa",
    ],
    "Royalty": [
        "king", "queen", "prince", "princess", "monarch", "ruler",
        "throne", "crown", "noble", "lord", "lady", "duke", "duchess",
        "emperor", "empress", "sultan", "tsar",
    ],
    "Gendered pairs": [
        "man", "woman", "male", "female", "masculine", "feminine",
        "boy", "girl", "gentleman", "lady", "husband", "wife",
        "he", "she", "his", "her", "himself", "herself",
    ],
    "Adjectives": [
        "strong", "gentle", "tough", "soft", "brave", "beautiful",
        "rational", "emotional", "aggressive", "nurturing", "assertive",
        "passive", "dominant", "submissive", "logical", "intuitive",
    ],
}

GENDER_ANCHORS = {
    "man": ["man", "male", "masculine"],
    "woman": ["woman", "female", "feminine"],
}
