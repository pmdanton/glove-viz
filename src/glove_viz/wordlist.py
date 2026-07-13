"""Curated word list — only these words are loaded from GloVe.

~150 words covering gender, jobs, family, royalty, adjectives, and
socioeconomic terms. Enough to illustrate embedding bias without
loading the full 400k-word vocabulary.
"""

WORDS: set[str] = {
    # ── Gender anchors ────────────────────────────────────────────────────
    "man", "woman", "male", "female", "masculine", "feminine",
    "boy", "girl", "gentleman", "lady", "he", "she", "his", "her",
    "himself", "herself", "men", "women", "husband", "wife",
    "father", "mother", "brother", "sister", "son", "daughter",
    "grandfather", "grandmother", "uncle", "aunt", "nephew", "niece",
    "king", "queen", "prince", "princess", "lord", "lady",
    "duke", "duchess", "emperor", "empress", "monarch", "ruler",

    # ── Jobs / professions ────────────────────────────────────────────────
    "surgeon", "doctor", "nurse", "teacher", "engineer", "programmer",
    "secretary", "soldier", "pilot", "ceo", "receptionist", "mechanic",
    "librarian", "chef", "dentist", "lawyer", "scientist", "astronaut",
    "firefighter", "babysitter", "architect", "cashier", "professor",
    "janitor", "housekeeper", "manager", "assistant", "driver", "farmer",
    "plumber", "electrician", "carpenter", "baker", "accountant",
    "banker", "analyst", "developer", "designer", "writer", "artist",
    "musician", "actor", "actress", "waiter", "waitress",
    "executive", "director", "officer", "clerk", "technician",
    "therapist", "psychologist", "researcher", "lecturer", "principal",

    # ── Socioeconomic / class ─────────────────────────────────────────────
    "educated", "uneducated", "manual", "skilled", "unskilled",
    "advanced", "basic", "rich", "poor", "wealthy", "destitute",
    "elite", "working", "upper", "lower", "middle",
    "privileged", "disadvantaged", "powerful", "powerless",

    # ── Adjectives (gendered stereotypes) ─────────────────────────────────
    "strong", "gentle", "tough", "soft", "brave", "beautiful",
    "rational", "emotional", "aggressive", "nurturing", "assertive",
    "passive", "dominant", "submissive", "logical", "intuitive",
    "competitive", "cooperative", "ambitious", "modest", "confident",
    "sensitive", "independent", "dependent", "decisive", "compassionate",

    # ── Race / ethnicity (for intersectional analysis) ────────────────────
    "black", "white", "asian", "african", "european", "american",
    "hispanic", "latino", "native", "immigrant", "citizen",

    # ── Misc useful ───────────────────────────────────────────────────────
    "smart", "dumb", "wise", "foolish", "leader", "follower",
    "boss", "employee", "winner", "loser", "success", "failure",
    "power", "weakness", "intelligence", "beauty",
}
