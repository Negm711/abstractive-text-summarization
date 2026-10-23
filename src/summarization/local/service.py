"""Reusable summarization service; no Gradio imports or model downloads."""
import time
from .prompts import POLICIES, REVIEW_INSTRUCTION, make_prompt
from .runtime import post

class SummaryError(ValueError):
    pass

class LocalSummarizer:
    def __init__(self, port=8087, policy="baseline", cache_prompt=True):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}")
        self.port, self.policy, self.cache_prompt = port, policy, cache_prompt

    def generate(self, instruction, content):
        prompt = make_prompt(instruction, content)
        tokens = post(self.port, "/tokenize", {"content": prompt, "parse_special": True})["tokens"]
        if len(tokens) + 160 > 4096:
            raise SummaryError("The article exceeds the local context limit for this mode. Nothing was truncated.")
        started = time.monotonic()
        result = post(self.port, "/completion", {
            "prompt": prompt, "n_predict": 160, "temperature": 0,
            "stop": ["<|im_end|>", "<|endoftext|>"], "stream": False,
            "cache_prompt": self.cache_prompt})
        answer = result.get("content", "").strip()
        if not answer or any(token in answer for token in ("<think>", "</think>", "<|im_")):
            raise SummaryError("No clean summary was returned. See the runtime log.")
        if result.get("stop_type") == "limit" or result.get("stopped_limit"):
            raise SummaryError("Generation reached its token limit. An incomplete summary was not accepted.")
        return {"summary": answer, "words": len(answer.split()), "input_tokens": len(tokens),
                "seconds": round(time.monotonic() - started, 3), "timings": result.get("timings", {})}

    def summarize(self, article, review=False, progress=None):
        if not isinstance(article, str) or not article.strip():
            raise SummaryError("Paste an English article first.")
        article = article.strip()
        if any(token in article for token in ("<|im_", "<|endoftext|>", "<think>", "</think>")):
            raise SummaryError("The article contains reserved model control tokens.")
        started = time.monotonic()
        if progress: progress(0.1, desc="Writing the summary…")
        first = self.generate(POLICIES[self.policy], "ARTICLE:\n" + article)
        final, reviewed = first, None
        note = "Direct summary · source review not requested."
        if review:
            if progress: progress(0.6, desc="Reviewing the draft against the source…")
            reviewed = self.generate(REVIEW_INSTRUCTION, "ARTICLE:\n" + article + "\n\nDRAFT SUMMARY:\n" + first["summary"])
            if reviewed["words"] <= 40:
                final = reviewed
                note = "Source review completed · check important claims before sharing."
            else:
                note = "Review exceeded the word target; the first draft was retained."
        if final["words"] > 40: note += " Summary exceeds the 40-word target."
        return {"summary": final["summary"], "draft": first["summary"], "note": note,
                "article_words": len(article.split()), "summary_words": final["words"],
                "total_seconds": round(time.monotonic() - started, 3),
                "mode": "source_review" if review else "direct", "policy": self.policy,
                "draft_timings": first["timings"], "review_timings": reviewed["timings"] if reviewed else None,
                "factual_accuracy_measured": False}
