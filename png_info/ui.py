import gradio as gr
from PIL import Image
import piexif
import os


class PngInfoUI:
    def __init__(self):
        pass

    def read_sd_parameters(self, image):
        """读取图片中的 Stable Diffusion 生成参数"""
        if image is None:
            return ""

        filename = os.path.basename(image).lower()
        parameters = ""

        if filename.endswith('.png'):
            try:
                with open(image, 'rb') as f:
                    png_data = f.read()

                idx = 8
                while idx < len(png_data) - 12:
                    chunk_length = int.from_bytes(png_data[idx:idx+4], 'big')
                    chunk_type = png_data[idx+4:idx+8]
                    chunk_content = png_data[idx+8:idx+8+chunk_length]

                    if chunk_type == b'tEXt':
                        try:
                            decoded = chunk_content.decode('utf-8')
                            if decoded.startswith('parameters\x00'):
                                parameters = decoded.split('\x00', 1)[1]
                                break
                        except:
                            pass

                    idx += chunk_length + 12
            except Exception as e:
                return f"Error reading PNG: {str(e)}"

        elif filename.endswith('.jpg') or filename.endswith('.jpeg'):
            try:
                img = Image.open(image)
                exif_dict = piexif.load(img.info.get('exif', b''))
                img.close()

                if 'Exif' in exif_dict and piexif.ImageIFD.UserComment in exif_dict['Exif']:
                    user_comment = exif_dict['Exif'][piexif.ImageIFD.UserComment]
                    if user_comment.startswith(b'UNICODE\x00'):
                        parameters = user_comment[8:].decode('utf-16')
                    else:
                        parameters = user_comment.decode('utf-8', errors='ignore')
            except Exception as e:
                return f"Error reading JPG: {str(e)}"

        if not parameters:
            parameters = "No Stable Diffusion parameters found in this image."

        return parameters

    def create_ui(self):
        with gr.Blocks(analytics_enabled=False) as interface:
            gr.HTML(value="""
            <h2>📷 PNG Info</h2>
            <p>Read Stable Diffusion generation parameters from images</p>
            """)

            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="filepath",
                        interactive=True
                    )
                    gr.Markdown("""
                    Supported formats:
                    - **PNG**: Reads parameters from tEXt chunk (key: parameters)
                    - **JPG**: Reads parameters from UserComment EXIF field
                    """)

                with gr.Column():
                    params_output = gr.Textbox(
                        label="Stable Diffusion Parameters",
                        lines=20,
                        max_lines=30,
                        interactive=False
                    )

            image_input.change(
                fn=self.read_sd_parameters,
                inputs=[image_input],
                outputs=[params_output]
            )

        return interface
