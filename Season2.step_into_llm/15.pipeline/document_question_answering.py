# 示例 1: 使用LayoutLMv2处理文档问答
from mindnlp.transformers import pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-layoutlmv2")

dqa_pipeline = pipeline("document-question-answering", model="hf-internal-testing/tiny-random-layoutlmv2",
                        tokenizer=tokenizer)

image_url = "https://pic.rmb.bdstatic.com/bjh/bc1fe8c9991/250405/6fc560e10aef1529da8d13fb25fc1c7f.jpeg"
question = "How many cats are there?"

outputs = dqa_pipeline(image=image_url, question=question, top_k=2)

print(outputs)


# 示例 2: 使用LayoutLM模型和自定义图像处理
from PIL import Image
import pytesseract
from mindnlp.transformers import pipeline, AutoTokenizer


def process_image_and_ocr(image_path):
    image = Image.open(image_path)
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = ocr_result['text']
    boxes = [ocr_result['left'], ocr_result['top'], ocr_result['width'], ocr_result['height']]
    return words, boxes


tokenizer = AutoTokenizer.from_pretrained("tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa", revision="9977165")

dqa_pipeline = pipeline("document-question-answering", model="tiennvcs/layoutlmv2-base-uncased-finetuned-docvqa",
                        tokenizer=tokenizer)

image_path = "invoice.png"
words, boxes = process_image_and_ocr(image_path)
question = "What is the invoice number?"

outputs = dqa_pipeline(question=question, words=words, boxes=boxes, top_k=2)

print(outputs)
