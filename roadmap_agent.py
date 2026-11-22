import json
import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from pdf_generator import create_roadmap_pdf


# ============================================================
# NERI — ADAPTIVE LEARNING ROADMAP AGENT
# ============================================================


class RoadmapState(TypedDict):
    feedback: dict
    duration_weeks: int
    skill_gaps: list
    roadmap: dict
    validation_errors: list
    attempts: int
    pdf_path: str
    status: str


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(content):
    """
    Safely parse JSON returned by the LLM.

    Handles:
    - Markdown code fences
    - Extra text before/after JSON
    - List-style response content
    """

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))

        content = "".join(parts)

    text = str(content).strip()

    # Remove markdown fences
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Remove possible "json" at the beginning
    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Find first JSON object
    start = text.find("{")

    # Find last JSON object bracket
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        print("\n--- RAW MODEL RESPONSE ---")
        print(text)
        print("--- END MODEL RESPONSE ---")

        raise ValueError("No valid JSON object found in model response.")

    text = text[start:end + 1]

    try:
        result = json.loads(text)

    except json.JSONDecodeError as exc:

        print("\n--- INVALID MODEL JSON ---")
        print(text)
        print("--- END INVALID JSON ---\n")

        raise ValueError(
            f"Invalid JSON from model: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result


# ============================================================
# BUILD GROQ LLM
# ============================================================

def build_llm() -> ChatGroq:

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from .env"
        )

    if not model:
        raise ValueError(
            "GROQ_MODEL is missing from .env"
        )

    print("\nUsing Groq model:", model)

    return ChatGroq(
        model=model,
        temperature=0.1,
        api_key=api_key
    )


# ============================================================
# NODE 1 — ANALYZE FEEDBACK
# ============================================================

def analyze_feedback(state: RoadmapState, llm: ChatGroq) -> dict:

    feedback = state["feedback"]

    prompt = f"""
Analyze this interview feedback.

Role:
{feedback["role"]}

Scores:
Overall: {feedback["overall_score"]}
Technical: {feedback["technical_score"]}
Communication: {feedback["communication_score"]}
HR: {feedback["hr_score"]}

Strengths:
{json.dumps(feedback["strengths"])}

Weaknesses:
{json.dumps(feedback["weaknesses"])}

Recommendations:
{json.dumps(feedback["recommendations"])}

Identify the most important 3 to 6 skill gaps.

Return ONLY JSON.

Required format:

{{
  "skill_gaps": [
    "skill gap 1",
    "skill gap 2",
    "skill gap 3"
  ]
}}

No Markdown.
No explanation.
"""

    response = llm.invoke(prompt)

    result = parse_json_response(response.content)

    gaps = result.get("skill_gaps")

    if not isinstance(gaps, list) or not gaps:

        raise ValueError(
            "Analyzer returned no skill gaps."
        )

    print("\n----------------------------------------")
    print("PRIORITIZED SKILL GAPS")
    print("----------------------------------------")

    for gap in gaps:
        print("-", gap)

    return {
        "skill_gaps": gaps
    }


# ============================================================
# NODE 2 — GENERATE ROADMAP
# ============================================================

def generate_roadmap(state: RoadmapState, llm: ChatGroq) -> dict:

    duration = state["duration_weeks"]
    feedback = state["feedback"]
    gaps = state["skill_gaps"]

    role = feedback["role"]

    prompt = f"""
Create a personalized learning roadmap.

Role:
{role}

Skill gaps:
{json.dumps(gaps)}

Duration:
{duration} weeks

Daily study time:
2 hours.

IMPORTANT OUTPUT RULES:

1. Return ONLY one valid JSON object.
2. Do not use Markdown.
3. Do not use ``` fences.
4. Do not add explanations.
5. Create exactly {duration} weeks.
6. Week numbers must be 1 to {duration}.
7. Each week must contain exactly 4 tasks.
8. Keep every task short.
9. Keep every text field short.
10. Do not use double quotes inside string values.
11. Make sure every string is closed.
12. Make sure every JSON object is closed.
13. Make sure every JSON array is closed.
14. Use correct commas.

Use this exact structure:

{{
  "role": "{role}",
  "duration_weeks": {duration},
  "main_gaps": [
    "gap 1",
    "gap 2",
    "gap 3"
  ],
  "weeks": [
    {{
      "week": 1,
      "objective": "Short objective",
      "focus_area": "Main topic",
      "tasks": [
        "Short task 1",
        "Short task 2",
        "Short task 3",
        "Short task 4"
      ],
      "practice_exercise": "Short practical exercise",
      "milestone": "Short measurable milestone"
    }}
  ],
  "final_goal": "Short final learning outcome"
}}

The weeks array MUST contain exactly {duration} objects.

Return the COMPLETE JSON object.
"""

    # --------------------------------------------------------
    # Try generating JSON twice
    # --------------------------------------------------------

    for attempt in range(2):

        print(
            f"\nGenerating roadmap... attempt {attempt + 1}"
        )

        response = llm.invoke(prompt)

        try:

            roadmap = parse_json_response(
                response.content
            )

            print("\n----------------------------------------")
            print("ROADMAP GENERATED SUCCESSFULLY")
            print("----------------------------------------")

            return {
                "roadmap": roadmap
            }

        except ValueError as error:

            print(
                f"\nRoadmap JSON attempt {attempt + 1} failed."
            )

            if attempt == 0:

                print(
                    "Retrying roadmap generation..."
                )

                prompt = f"""
Generate the learning roadmap again.

Role:
{role}

Skill gaps:
{json.dumps(gaps)}

Duration:
{duration} weeks

Return ONLY valid JSON.

Required top-level fields:

role
duration_weeks
main_gaps
weeks
final_goal

Every week must contain:

week
objective
focus_area
tasks
practice_exercise
milestone

STRICT RULES:

- Exactly {duration} weeks
- Exactly 4 tasks per week
- Week numbers from 1 to {duration}
- No Markdown
- No explanations
- No comments
- No text outside JSON
- Close every string
- Close every object
- Close every array
- Use valid commas
- Keep all text short

Return the complete JSON object.
"""

            else:

                raise error


# ============================================================
# NODE 3 — VALIDATE ROADMAP
# ============================================================

def validate_roadmap(state: RoadmapState) -> dict:

    roadmap = state["roadmap"]

    expected = state["duration_weeks"]

    errors = []

    # Check duration
    if roadmap.get("duration_weeks") != expected:

        errors.append(
            "Roadmap duration does not match selected duration."
        )

    # Check role
    if not roadmap.get("role"):

        errors.append(
            "Missing target role."
        )

    # Check main gaps
    if (
        not isinstance(
            roadmap.get("main_gaps"),
            list
        )
        or not roadmap.get("main_gaps")
    ):

        errors.append(
            "Missing main_gaps list."
        )

    # Check weeks
    weeks = roadmap.get("weeks")

    if not isinstance(weeks, list):

        errors.append(
            "weeks must be a list."
        )

        weeks = []

    elif len(weeks) != expected:

        errors.append(
            f"Expected {expected} weeks; received {len(weeks)}."
        )

    # Required fields
    required_fields = [
        "objective",
        "focus_area",
        "tasks",
        "practice_exercise",
        "milestone"
    ]

    # Validate each week
    for index, week in enumerate(
        weeks,
        start=1
    ):

        if not isinstance(week, dict):

            errors.append(
                f"Week {index} must be an object."
            )

            continue

        # Check numbering
        if week.get("week") != index:

            errors.append(
                f"Week {index} has incorrect numbering."
            )

        # Check required fields
        for field in required_fields:

            if not week.get(field):

                errors.append(
                    f"Week {index} is missing {field}."
                )

        # Check tasks
        tasks = week.get("tasks")

        if not isinstance(tasks, list):

            errors.append(
                f"Week {index} tasks must be a list."
            )

        elif len(tasks) != 4:

            errors.append(
                f"Week {index} must contain exactly 4 tasks."
            )

    # Final goal
    if not roadmap.get("final_goal"):

        errors.append(
            "Missing final_goal."
        )

    # Print validation result
    print("\n----------------------------------------")

    if errors:

        print("ROADMAP VALIDATION: FAILED")

        for error in errors:
            print("-", error)

    else:

        print("ROADMAP VALIDATION: PASSED")

    print("----------------------------------------")

    return {
        "validation_errors": errors
    }


# ============================================================
# NODE 4 — REVISE ROADMAP
# ============================================================

def revise_roadmap(
    state: RoadmapState,
    llm: ChatGroq
) -> dict:

    roadmap = state["roadmap"]

    errors = state["validation_errors"]

    duration = state["duration_weeks"]

    prompt = f"""
Fix the following learning roadmap.

Current roadmap:
{json.dumps(roadmap, indent=2)}

Validation errors:
{json.dumps(errors, indent=2)}

Required duration:
{duration} weeks

Fix ALL validation errors.

Return ONLY valid JSON.

Required fields:

role
duration_weeks
main_gaps
weeks
final_goal

Every week must contain:

week
objective
focus_area
tasks
practice_exercise
milestone

Rules:

- Exactly {duration} weeks
- Week numbers 1 to {duration}
- Exactly 4 tasks per week
- No Markdown
- No explanations
- No comments
- No text outside JSON
- Preserve useful existing content
"""

    response = llm.invoke(prompt)

    revised = parse_json_response(
        response.content
    )

    new_attempts = state["attempts"] + 1

    print(
        f"\nRoadmap revision attempt: {new_attempts}"
    )

    return {
        "roadmap": revised,
        "attempts": new_attempts
    }


# ============================================================
# NODE 5 — EXPORT PDF
# ============================================================

def export_pdf(state: RoadmapState) -> dict:

    if state["validation_errors"]:

        raise ValueError(
            "Cannot export an invalid roadmap."
        )

    duration = state["duration_weeks"]

    filename = (
        f"NERI_Roadmap_{duration}_Weeks.pdf"
    )

    output_directory = "generated_roadmaps"

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_path = os.path.join(
        output_directory,
        filename
    )

    create_roadmap_pdf(
        state["roadmap"],
        output_path
    )

    print("\n========================================")
    print("PDF CREATED SUCCESSFULLY")
    print("========================================")
    print("File:", output_path)

    return {
        "pdf_path": output_path,
        "status": "completed"
    }


# ============================================================
# NODE 6 — FAILURE
# ============================================================

def mark_failed(
    state: RoadmapState
) -> dict:

    print("\n========================================")
    print("ROADMAP GENERATION FAILED")
    print("========================================")

    print(
        "The roadmap could not pass validation."
    )

    return {
        "status": "failed"
    }


# ============================================================
# GRAPH ROUTING
# ============================================================

MAX_REVISION_ATTEMPTS = 2


def route_after_validation(
    state: RoadmapState
) -> str:

    # If valid → export
    if not state["validation_errors"]:

        return "export"

    # If invalid but revisions remain
    if state["attempts"] < MAX_REVISION_ATTEMPTS:

        return "revise"

    # Otherwise fail
    return "failed"


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def build_graph(
    llm: ChatGroq
):

    builder = StateGraph(
        RoadmapState
    )

    # Add nodes
    builder.add_node(
        "analyze_feedback",
        lambda state:
        analyze_feedback(
            state,
            llm
        )
    )

    builder.add_node(
        "generate_roadmap",
        lambda state:
        generate_roadmap(
            state,
            llm
        )
    )

    builder.add_node(
        "validate_roadmap",
        validate_roadmap
    )

    builder.add_node(
        "revise_roadmap",
        lambda state:
        revise_roadmap(
            state,
            llm
        )
    )

    builder.add_node(
        "export_pdf",
        export_pdf
    )

    builder.add_node(
        "mark_failed",
        mark_failed
    )

    # --------------------------------------------------------
    # Graph flow
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "analyze_feedback"
    )

    builder.add_edge(
        "analyze_feedback",
        "generate_roadmap"
    )

    builder.add_edge(
        "generate_roadmap",
        "validate_roadmap"
    )

    # Validation routing
    builder.add_conditional_edges(
        "validate_roadmap",
        route_after_validation,
        {
            "revise": "revise_roadmap",
            "export": "export_pdf",
            "failed": "mark_failed"
        }
    )

    # Revision goes back to validation
    builder.add_edge(
        "revise_roadmap",
        "validate_roadmap"
    )

    # Finish
    builder.add_edge(
        "export_pdf",
        END
    )

    builder.add_edge(
        "mark_failed",
        END
    )

    return builder.compile()