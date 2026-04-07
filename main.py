import sys
from agent.orchestrator import OrchestratorAgent


def main():
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter research topic: ").strip()

    if not topic:
        print("No topic provided.")
        return

    orchestrator = OrchestratorAgent()
    orchestrator.run(topic)
    print("\n" + "=" * 60)
    print("Report saved to report.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
