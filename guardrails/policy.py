"""Yes/no policy questions for the input guardrail.

Each question is phrased so that ``noul`` (``P(yes)``) is the confidence that
the user's query passes the check. The ``state`` is a JSON object with the
previous conversation and the user's latest message.
"""

from __future__ import annotations

LEGAL_DOMAIN = "legal"
NO_SELF_INCRIMINATION = "no_self_incrimination"

LEGAL_DOMAIN_QUESTION = """\
The state is a JSON object with two fields: "conversation" (previous turns
between the user and the legal assistant, possibly an empty list) and "message"
(the user's latest message, which is the one to evaluate). Answer YES only if
"message" is a request for legal information or legal advice, or a follow-up
answer that continues a legal consultation visible in "conversation": rights,
obligations, contracts, liability, procedures, penalties, or the legal
consequences of a situation under the law. Short answers such as "Sí", "No",
"No estoy casado" or "Llevo tres años" are YES when they answer a question
about a legal consultation. Answer NO if "message" is unrelated to law
(cooking, programming, medicine, shopping, personal chat) or if it only
contains an incidental legal word without asking a legal question.
"""

NO_SELF_INCRIMINATION_QUESTION = """\
The state is a JSON object with "conversation" (previous turns, possibly empty)
and "message" (the user's latest message, which is the one to evaluate). Answer
YES only if "message" does NOT present the user as the author, participant or
planner of a criminal offence, and does NOT ask for help to commit, conceal or
evade a crime. Use "conversation" to interpret short answers: "Sí" or "No" is
YES when it answers a question about a general legal scenario, and NO when it
admits that the user committed or planned a crime. Answer YES for general legal
questions, questions about third parties, and first-person statements about
being a victim or about non-criminal civil, family or labour matters. Answer
YES for hypothetical questions too, including questions phrased as "¿qué pasa
si...?", "si yo...", or "¿qué consecuencias tendría...?", because a
hypothetical does not assert that the user committed the fact. For example,
"¿Qué pasa si no pago la pensión de alimentos?" is a hypothetical and must be
YES. Answer NO if the user says or implies "I did", "I am doing", "I omitted"
or "I plan to do" something that may constitute a crime, even if the question
is about the penalty, defence or consequences; for example, "No le pago la
pensión a mis hijos" is an admission and must be NO. Also answer NO if the user
asks how to commit, conceal or evade a crime.
"""

CHECKS: dict[str, str] = {
    LEGAL_DOMAIN: LEGAL_DOMAIN_QUESTION,
    NO_SELF_INCRIMINATION: NO_SELF_INCRIMINATION_QUESTION,
}
