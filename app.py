import json
from pathlib import Path

from roadmap_agent import RoadmapState, build_graph, build_llm


INTERVIEW_FEEDBACK = {
    "role": "Junior AI Engineer",
    "overall_score": 62,
    "technical_score": 70,
    "communication_score": 45,
    "hr_score": 65,
    "strengths": [
        "Good Python fundamentals",
        "Understands basic machine learning",
        "Has implemented small ML projects"
    ],
    "weaknesses": [
        "Unable to explain projects clearly",
        "Weak communication structure",
        "Limited knowledge of model deployment",
        "Needs more practical ML experience"
    ],
    "recommendations": [
        "Practice explaining technical concepts",
        "Revise ML algorithms",
        "Learn model deployment using FastAPI",
        "Build practical ML projects"
    ]
}


def choose_duration() -> int:
    options = {"1": 2, "2": 4, "3": 8}

    print("\nSelect roadmap duration:")
    print("1. 2 weeks")
    print("2. 4 weeks")
    print("3. 8 weeks")

    choice = input("Enter 1, 2 or 3: ").strip()

    if choice not in options:
        raise ValueError("Invalid duration selection. Choose 1, 2 or 3.")

    return options[choice]


def main():
    print("=" * 58)
    print("NERI — Adaptive Learning Roadmap Agent")
    print("=" * 58)

    llm = build_llm()

    test = llm.invoke(
        "Reply with exactly: Neri roadmap agent connection successful."
    )
    print("\n" + str(test.content))

    print("\nFeedback prepared for:", INTERVIEW_FEEDBACK["role"])

    duration_weeks = choose_duration()
    print(f"Selected duration: {duration_weeks} weeks")

    initial_state: RoadmapState = {
        "feedback": INTERVIEW_FEEDBACK,
        "duration_weeks": duration_weeks,
        "skill_gaps": [],
        "roadmap": {},
        "validation_errors": [],
        "attempts": 0,
        "pdf_path": "",
        "status": "processing"
    }

    graph = build_graph(llm)

    print("\nStarting LangGraph workflow...")
    result = graph.invoke(initial_state)

    print("\nFinal status:", result["status"])

    if result["status"] == "completed":
        print("PDF:", result["pdf_path"])
        print("\nFinal roadmap:")
        print(json.dumps(result["roadmap"], indent=2))

        pdf = Path(result["pdf_path"])
        assert pdf.exists(), "PDF was not created."
        print(f"\nPDF size: {pdf.stat().st_size} bytes")
    else:
        print("Validation errors:", result["validation_errors"])


if __name__ == "__main__":
    main()
