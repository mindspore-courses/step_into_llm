import os
from logic import add_text, generate_response, render_file, clear_chatbot

import gradio as gr

# Gradio application setup
def create_demo():
    with gr.Blocks(title= " PDF Chatbot",
        theme = "Soft"  # Change the theme here
        ) as demo:
        
        # Create a Gradio block

        with gr.Column():
            with gr.Row():
                chatbot = gr.Chatbot(value=[], elem_id='chatbot', height=600)
                show_img = gr.Image(label='PDF Preview', height=600)

        with gr.Row():
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Ask your pdf?",
                container=False,
                render=False)
            gr.Examples(["这篇论文试图解决什么问题？", "有哪些相关研究？",
                         "论文如何解决这个问题？", "论文做了哪些实验？",
                         "有什么可以进一步探索的点？", "总结一下本文的主要内容"], text_input)

        with gr.Row():
            with gr.Column(scale=0.60):
                text_input.render()

            with gr.Column(scale=0.20):
                submit_btn = gr.Button('Send')

            with gr.Column(scale=0.20):
                upload_btn = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])


        return demo, chatbot, show_img, text_input, submit_btn, upload_btn

demo, chatbot, show_img, txt, submit_btn, btn = create_demo()

# Set up event handlers
with demo:
    # Event handler for uploading a PDF
    btn.upload(render_file, inputs=[btn], outputs=[show_img]).success(clear_chatbot, outputs=[chatbot])

    # Event handler for submitting text and generating response
    submit_btn.click(add_text, inputs=[chatbot, txt], outputs=[chatbot], queue=False).\
        success(generate_response, inputs=[chatbot, txt, btn], outputs=[chatbot, txt])
if __name__ == "__main__":
    demo.launch()
