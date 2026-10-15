"""Customer wording for unresolved replies; never a financial classifier.

These prompts ask for missing meaning or explain the current premise. They
produce no bucket, option, fact or replacement for an abstaining ML model.
An identifiable D5 gap is recognised only to choose the question to ask.
"""
from __future__ import annotations

import re

from ai.question_spec import BusinessSpec


EXPLANATIONS = {
    "D1": "This asks what you expect to use the same money for. There can be one purpose or more than one.",
    "D2": "This optional question asks for your current age range. You may leave it unanswered. Age does not determine your risk preference, loss capacity or time horizon.",
    "D3": "This is the amount of existing money we are discussing, rather than a future savings target. A range and the currency are enough; you may leave the amount unanswered.",
    "D4": "This asks where the money we are discussing is held now. Money held inside a pension is different from pension payments you receive. Whether money can be accessed is checked separately.",
    "D5": "This means what pays your housing, food and bills now. You can name one source or several. Benefits and regular support from someone else share one answer category. Money merely held, expected in future or used for someone else's costs does not automatically count.",
    "D6": "Imagine the money we are discussing was unavailable for the whole of the next 12 months. Consider other income or money that could pay essential costs and required repayments when due. This asks whether all, some or none would be covered.",
    "D7": "This is about repayments currently required on loans or credit: whether there are none, or you are keeping up easily, with difficulty, or falling behind. Having debt does not always mean a repayment is required now.",
    "D8": "Personally held means you owned the investment yourself. Reading about investments or following someone else's investments does not establish what you have held. You can name more than one type.",
    "D9": "Count investment purchases or sales you made or instructed in the past 12 months, including recurring orders that actually executed. Exclude ordinary bank transfers and trades a manager chose independently. An unknown count is different from none.",
    "D10": "Count fresh money that actually went into investments during the past 12 months, including invested pension contributions once. Exclude investment growth and money reinvested from sales. A range and the currency are enough; this is optional.",
    "D11": "Market investments can fall in value. Some or all of the original money could be permanently lost, and recovery is not guaranteed. This question checks your understanding of that possibility.",
    "D12": "This means the earliest point when you might realistically need any of the same money. A preferred investment goal date can be later than a possible need for living costs or emergencies.",
    "D13": "Imagine losing 20% of this money with no guaranteed recovery. Consider essential costs, how you pay them, your budget and other plans. This is one example of practical impact, not your maximum affordable loss.",
    "D14": "This asks which trade-off you prefer between possible growth, ups and downs and possible losses. Deliberately choosing a balance is different from being unsure. Your preference is separate from what losses you could afford.",
    "D15": "For this feelings question, imagine a 20% fall while essential costs and known plans stay covered. Recovery is not guaranteed. Consider whether you could accept the situation emotionally, with or without worry.",
}


D5_COMPLETE_FUNDING_PROMPT = (
    "The earlier reply left some living-cost funding unresolved. Before I can confirm a whole answer, "
    "please describe what currently pays both your housing and your other living costs "
    "in one reply. If the earlier description was wrong, you can correct it."
)


NEUTRAL_FOLLOWUPS = {
    "D1": "What would you expect to use this same money for? You can describe one purpose or several, or leave it unresolved.",
    "D2": "Would you like to give your current age range, use an age-band choice, or leave this optional question unanswered?",
    "D3": "What existing amount or range, with its currency, are we discussing? You can leave this optional amount unanswered.",
    "D4": "Where is the money we are discussing held now? You can describe one place or several, use a choice, or pause.",
    "D5": "I haven't understood enough to match an answer yet. You can describe what pays these costs now, use one of the choices, or pause.",
    "D6": "Excluding this money, would your other resources cover all, some or none of the essential costs and required repayments when due throughout those 12 months? It can remain unresolved.",
    "D7": "Are any loan or credit repayments required now, and if so are you keeping up easily, with difficulty, or falling behind? You can describe your situation or leave it unresolved.",
    "D8": "Which investment types, if any, have you personally held? You can describe them, use a choice, or leave this optional question unanswered.",
    "D9": "How many actual investment purchases or sales did you make or instruct in the past 12 months? You can use a count band or leave the count unknown.",
    "D10": "How much fresh money was actually invested in the past 12 months, and in which currency? You can give a range or leave this optional question unanswered.",
    "D11": "In your own words, what could happen to the original money, and is recovery guaranteed? It is fine to ask for an explanation or pause.",
    "D12": "What is the earliest point when you might realistically need any of this same money, including possible earlier needs? It can remain unresolved.",
    "D13": "For that same 20% loss, would essential costs become difficult, would your funding, budget or plans change, or would all of those be unaffected? You can describe the impact or leave it unresolved.",
    "D14": "Which trade-off do you prefer between possible growth, ups and downs and possible loss? Being unsure is a valid unresolved reply; it does not mean choosing the balanced option.",
    "D15": "With essential costs and known plans covered in this example, would the worry make the situation unacceptable, acceptable while worried, or comfortable? You can remain unsure or pause.",
}


# Display-only progressions for questions with weak observed free-text
# coverage. An abstention does not establish a specific misunderstanding:
# explain the scope neutrally instead of inventing a missing customer fact.
# These messages change no model input, prediction, budget or recorded field.
REPEATED_FOLLOWUPS = {
    "D1": (
        "Think about what you expect this same money to pay for or help you achieve. "
        "There may be one purpose or several. What would you use it for? You can remain unsure.",
        "I still cannot match a supported purpose. You can describe it in your own words, "
        "choose an option if it fits, or leave it unresolved. If it remains unresolved, we will pause this question.",
    ),
    "D3": (
        "This question concerns money you already have, rather than an amount you hope to save. "
        "What amount or range, and currency, applies to that existing money? You may leave this optional question unanswered.",
        "I still cannot match an amount for the existing money. You can give a range and currency in your own words, "
        "choose a matching range, or leave this optional question unanswered. If it remains unresolved, we will pause this question.",
    ),
    "D10": (
        "For the past 12 months, count fresh money actually put into investments, including invested pension contributions once. "
        "Leave out growth and money reused from investment sales. What amount or range, and currency, applies? You may leave this optional question unanswered.",
        "I still cannot match the new contributions over those 12 months. You can describe the amount or range and currency, "
        "choose an option if it fits, or leave this optional question unanswered. If it remains unresolved, we will pause this question.",
    ),
    "D12": (
        "Your preferred goal date and the earliest realistic need can differ. When might you first need any of this same money "
        "for living costs, emergencies or another purpose? You can remain unsure.",
        "I still cannot establish the earliest realistic need for this same money. You can give a timeframe in your own words, "
        "choose an option if it fits, or leave it unresolved. If it remains unresolved, we will pause this question.",
    ),
    "D15": (
        "In this example, essential costs and known plans stay covered despite a 20% fall, and recovery is not guaranteed. "
        "Would it feel unacceptable, acceptable while worried, or comfortable? You can describe your feelings in your own words or remain unsure.",
        "I still cannot match how you would feel about that example. You can describe whether you could accept it emotionally, "
        "choose an option if it fits, or remain unsure. If it remains unresolved, we will pause this question.",
    ),
}


def question_explanation(spec: BusinessSpec, qid: str) -> str:
    """Useful display-only Help, without a new factual request or inference."""
    if qid in EXPLANATIONS:
        return EXPLANATIONS[qid]
    question = spec.question(qid)
    return getattr(question, "display_context", "") or question.question


def d5_focused_followup(text: str, reason: str = "") -> str | None:
    """Ask an explicitly identifiable gap without deciding a financial answer.

    General model uncertainty is insufficient evidence that ownership, timing
    or use is missing. Veto reasons identify those boundaries, or the customer
    explicitly describes the missing remainder. Nothing is inferred or saved.
    """
    normal = " ".join(text.lower().replace("’", "'").split())
    for clause in re.split(r"[.;!?]|\bbut\b", normal):
        if re.search(r"\b(?:next (?:week|month|year)|in the future|will (?:pay|fund|cover)|used to|last year)\b", clause):
            continue
        if re.search(
            r"\b(?:don't know|do not know|not sure|unsure|cannot tell|can't tell)\b"
            r".{0,65}\b(?:rest|remaining|other (?:costs|expenses|bills)|everything else)\b",
            clause,
        ):
            return "What currently pays for your other living costs, such as food and bills? Please include what pays your housing as well, so your reply describes the whole current funding. It can remain unresolved."
    if reason in {"candidate_withdrawal_ownership_or_use_conflict", "unsupported_ownership"}:
        return "Is that money paying your living costs, or someone else's? Please describe whose costs and which source you mean."
    if reason == "future_income_not_current":
        return "What currently pays your living costs now, rather than an expected future payment? You can name the source or leave it unresolved."
    if reason == "holding_not_income":
        return "Do you mean pension payments you receive now, or money still held inside a pension? Please also say what currently pays your living costs."
    if reason == "current_source_not_established":
        return "Does that money currently pay any of your everyday living costs? Please name the source and say whether you use it for those costs."
    if reason in {"missing_source_detail", "named_sources_for_multiple_missing"}:
        return "Which source or sources currently pay your living costs? Please name them; you do not need to choose a category yourself."
    return None


def d5_reply_covers_whole_funding_scope(text: str) -> bool:
    """Scope check only: a reply to a part-question must cover the whole question.

    This cannot establish contributors or an option. The unchanged ML and
    business checks must still interpret the original reply. A complete new
    statement can correct the earlier account; fragmentary corrections cannot
    establish how the other costs are paid.
    """
    normal = " ".join(text.lower().replace("’", "'").split())
    if re.search(
        r"\b(?:all|every|entire) (?:of )?(?:my |our |the )?"
        r"(?:(?:current|everyday|regular|daily|living|essential|household) )*"
        r"(?:costs|expenses|bills)\b|\b(?:everything|all expenses)\b", normal,
    ):
        # 'Everything else' still names only the remainder; housing must also
        # appear below. 'Everything is...' or 'pays everything' names the whole.
        if not re.search(r"\beverything else\b", normal) or re.search(
                r"\b(?:all|every|entire)\b.{0,45}\b(?:costs|expenses|bills)\b", normal):
            return True
    housing = re.search(r"\b(?:rent|housing|mortgage|accommodation)\b", normal)
    remainder = re.search(r"\b(?:food|groceries|bills|remaining (?:costs|expenses)|the rest|everything else|other living costs)\b", normal)
    return bool(housing and remainder)


def unresolved_followup(qid: str, text: str, reason: str, *, request_number: int = 1,
                        model_uncertain: bool = False, bucket: str | None = None) -> str:
    """A neutral question tied to this question's meaning, never a classification."""
    if qid == "D5":
        focused = d5_focused_followup(text, reason)
        if focused:
            return focused
        if request_number == 2:
            return "For this question, name what pays your housing, food and bills now. Receiving or holding money alone does not tell me whether it pays those costs. You can use a choice, leave this unresolved or pause."
        if request_number >= 3:
            return "I still cannot match a supported answer. Please describe your current living-cost funding in one reply, or use a choice. If it remains unclear, we will pause this question."
    prompt = (REPEATED_FOLLOWUPS[qid][min(request_number, 3) - 2]
              if qid in REPEATED_FOLLOWUPS and request_number >= 2
              else NEUTRAL_FOLLOWUPS[qid])
    if bucket == "Confusion" and not model_uncertain:
        return "Let's return to this question. " + prompt
    if model_uncertain and qid != "D5":
        return "I haven't understood enough to match an answer yet. " + prompt
    return prompt
