INSTRUCTION = (
    "Summarize the article in English in one or two complete sentences, using at most 40 words. "
    "State the main event and its most important outcome or context. Use only information "
    "supported by the article. Preserve uncertainty, attribution, names, numbers and who did "
    "what to whom. Do not invent missing details. Write a concise, natural paraphrase without "
    "commentary, a heading, reasoning, or a list."
)

GROUNDED_INSTRUCTION = (
    "Write a concise English summary of the article in one or two complete sentences, at most 40 words. "
    "Lead with its central event or finding and most important outcome. Every assertion must be "
    "supported by the article. Keep proposed, alleged, ongoing and completed events distinct. "
    "Preserve uncertainty: could or may must not become will or did. Do not invent a denial, "
    "intention, causal link, identity or injury. Keep who did what to whom clear. "
    "Combine related facts in natural language without ambiguous chains of clauses. "
    "Omit secondary schedules, quotations and background when they crowd out the main point. "
    "Use names and numbers only when supported and useful. Return only the summary, "
    "without a title, list, commentary or reasoning."
)
REVIEW_INSTRUCTION = (
    "Edit the draft summary using only the article. Preserve its main event and most important "
    "outcome or practical consequence. Correct unsupported claims, reversed roles, numbers, negation "
    "and uncertainty. Remove invented details. Keep accurate content when no correction is needed. "
    "Return only the final natural English summary, at most 40 words and one or two complete sentences. "
    "Do not output explanations, reasoning or a heading."
)
POLICIES = {"baseline": INSTRUCTION, "grounded-v1": GROUNDED_INSTRUCTION}


def make_prompt(instruction, content):
    return (f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")
