"""Start the Gradio web app."""

from traffic_law_rag.web import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
