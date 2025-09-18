import re
from typing import Tuple

# Define simple irreversible state patterns and forbidden reversals
IRREVERSIBLE = [
    # (establish_regex, forbidden_later_regex)
    (r"\b(shatter(ed|s)?|broke(n)?|crack(ed)?)\b.*\b(mug|glass|bottle|pottery|cup)\b",
     r"\b(unbroken|whole again|as good as new|intact again|fixed perfectly|no longer broken)\b"),

    (r"\b(crack(ed)?|cooked|fried|scrambl(ed|e))\b.*\b(egg|yolk)\b",
     r"\b(raw again|uncooked|shell reformed|uncracked)\b"),

    (r"\b(remove(d)?|excis(ed)?|amputat(ed)?)\b.*\b(appendix|limb|organ)\b",
     r"\b(grew back|restored physically|never removed)\b"),

    (r"\b(melt(ed|s)?|thaw(ed)?)\b.*\b(ice|sculpture)\b",
     r"\b(refroze(d)? into shape|restored sculpture|back to original form)\b"),

    (r"\b(burn(ed|t)?|incinerat(ed)?)\b.*\b(letter|paper|document)\b",
     r"\b(not burned|intact again|unburned|restored perfectly)\b"),

    (r"\b(cut down|fell(ed)?|chop(ped)?)\b.*\b(tree|forest)\b",
     r"\b(regrew instantly|back to full size|never cut)\b"),

    (r"\b(void(ed)?|destroy(ed)?|cancel(led)?)\b.*\b(contract|agreement)\b",
     r"\b(valid again|restored|never voided)\b"),

    (r"\b(die(d)?|pass(ed)? away|deceas(ed)?)\b",
     r"\b(came back to life|alive again|resurrected|never died)\b")
]

def find_contradiction(text: str) -> Tuple[bool, str]:
    """Check if text contains thermodynamic contradiction via regex rules."""
    text_lower = text.lower()
    
    for establish_pattern, forbidden_pattern in IRREVERSIBLE:
        # Look for the establishment of irreversible state
        if re.search(establish_pattern, text_lower, re.IGNORECASE):
            # Then check if there's a forbidden reversal later
            if re.search(forbidden_pattern, text_lower, re.IGNORECASE):
                return True, f"Irreversible process reversed: {establish_pattern[:30]}... -> {forbidden_pattern[:30]}..."
    
    return False, "No contradiction detected"
