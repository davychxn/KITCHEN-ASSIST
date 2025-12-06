"""
Create a neat workflow diagram PowerPoint presentation
for the Pan/Pot State Detection System
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


def add_rounded_rectangle(slide, left, top, width, height, text, fill_color, text_color=(255, 255, 255)):
    """Add a rounded rectangle shape with text"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        left, top, width, height
    )
    
    # Set fill color
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(*fill_color)
    
    # Set border
    shape.line.color.rgb = RGBColor(80, 80, 80)
    shape.line.width = Pt(1.5)
    
    # Add text
    text_frame = shape.text_frame
    text_frame.text = text
    text_frame.word_wrap = True
    text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    
    # Format text
    for paragraph in text_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(*text_color)
    
    return shape


def add_arrow(slide, x1, y1, x2, y2, label=""):
    """Add an arrow connector between two points"""
    connector = slide.shapes.add_connector(
        1,  # Straight connector
        x1, y1, x2, y2
    )
    connector.line.color.rgb = RGBColor(80, 80, 80)
    connector.line.width = Pt(2)
    
    # Add arrow end
    connector.line.end_arrow_type = 2
    
    # Add label if provided
    if label:
        # Calculate midpoint
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        
        # Add text box for label
        textbox = slide.shapes.add_textbox(
            mid_x - Inches(0.5), mid_y - Inches(0.2),
            Inches(1), Inches(0.3)
        )
        text_frame = textbox.text_frame
        text_frame.text = label
        text_frame.word_wrap = False
        
        for paragraph in text_frame.paragraphs:
            paragraph.alignment = PP_ALIGN.CENTER
            for run in paragraph.runs:
                run.font.size = Pt(11)
                run.font.color.rgb = RGBColor(60, 60, 60)
    
    return connector


def create_workflow_presentation():
    """Create the complete workflow presentation"""
    
    prs = Presentation()
    prs.slide_width = Inches(13.33)  # Widescreen
    prs.slide_height = Inches(7.5)
    
    # ==================== Slide 1: Title Slide ====================
    title_slide_layout = prs.slide_layouts[6]  # Blank layout
    slide1 = prs.slides.add_slide(title_slide_layout)
    
    # Background color
    background = slide1.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(240, 248, 255)  # Light blue
    
    # Title
    title_box = slide1.shapes.add_textbox(
        Inches(1), Inches(2.5), Inches(11.33), Inches(1)
    )
    title_frame = title_box.text_frame
    title_frame.text = "Pan/Pot State Detection System"
    for paragraph in title_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(48)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
    
    # Subtitle
    subtitle_box = slide1.shapes.add_textbox(
        Inches(1), Inches(3.8), Inches(11.33), Inches(0.6)
    )
    subtitle_frame = subtitle_box.text_frame
    subtitle_frame.text = "AI-Powered Kitchen Safety Workflow"
    for paragraph in subtitle_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(24)
            run.font.color.rgb = RGBColor(100, 100, 100)
    
    # ==================== Slide 2: Overall Workflow ====================
    slide2 = prs.slides.add_slide(title_slide_layout)
    
    # Background
    background = slide2.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(250, 250, 250)
    
    # Title
    title_box = slide2.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.6)
    )
    title_frame = title_box.text_frame
    title_frame.text = "System Workflow Overview"
    for paragraph in title_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(32)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
    
    # Define colors
    color_input = (100, 149, 237)      # Cornflower blue
    color_detect = (255, 165, 0)       # Orange
    color_classify = (50, 205, 50)     # Lime green
    color_output = (220, 20, 60)       # Crimson
    
    # Box dimensions
    box_width = Inches(2.2)
    box_height = Inches(1)
    y_start = Inches(2)
    x_spacing = Inches(2.8)
    
    # Step 1: Input Image
    x1 = Inches(1)
    box1 = add_rounded_rectangle(
        slide2, x1, y_start, box_width, box_height,
        "Input Image\n(Kitchen Camera)", color_input
    )
    
    # Step 2: Object Detection
    x2 = x1 + box_width + x_spacing
    box2 = add_rounded_rectangle(
        slide2, x2, y_start, box_width, box_height,
        "Object Detection\n(YOLO v8)", color_detect
    )
    
    # Step 3: State Classification
    x3 = x2 + box_width + x_spacing
    box3 = add_rounded_rectangle(
        slide2, x3, y_start, box_width, box_height,
        "State Classification\n(ResNet50)", color_classify
    )
    
    # Step 4: Output
    x4 = x3 + box_width + x_spacing
    box4 = add_rounded_rectangle(
        slide2, x4, y_start, box_width, box_height,
        "Output Result\n(with Marking)", color_output
    )
    
    # Arrows
    add_arrow(slide2, 
              x1 + box_width, y_start + box_height/2,
              x2, y_start + box_height/2)
    
    add_arrow(slide2,
              x2 + box_width, y_start + box_height/2,
              x3, y_start + box_height/2)
    
    add_arrow(slide2,
              x3 + box_width, y_start + box_height/2,
              x4, y_start + box_height/2)
    
    # Add state boxes below
    y_states = Inches(4)
    state_width = Inches(1.5)
    state_height = Inches(0.6)
    
    # State colors
    state_colors = {
        'Normal': (0, 255, 0),
        'Boiling': (255, 255, 0),
        'Smoking': (128, 128, 128),
        'On Fire': (0, 0, 255)
    }
    
    states = list(state_colors.keys())
    x_state_start = Inches(2.5)
    state_spacing = Inches(2)
    
    for i, state in enumerate(states):
        x_state = x_state_start + i * state_spacing
        add_rounded_rectangle(
            slide2, x_state, y_states, state_width, state_height,
            state, state_colors[state], (0, 0, 0)
        )
    
    # Add "Detected States:" label
    label_box = slide2.shapes.add_textbox(
        Inches(0.5), y_states, Inches(1.8), state_height
    )
    label_frame = label_box.text_frame
    label_frame.text = "Detected States:"
    label_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    for paragraph in label_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = Pt(14)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 0, 0)
    
    # ==================== Slide 3: Detailed Processing ====================
    slide3 = prs.slides.add_slide(title_slide_layout)
    
    # Background
    background = slide3.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(250, 250, 250)
    
    # Title
    title_box = slide3.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.6)
    )
    title_frame = title_box.text_frame
    title_frame.text = "Detailed Processing Pipeline"
    for paragraph in title_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(32)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
    
    # Left column - Data Preparation
    y_pos = Inches(1.3)
    x_left = Inches(0.8)
    box_w = Inches(2.5)
    box_h = Inches(0.8)
    spacing = Inches(0.9)
    
    # Data Preparation Section
    section_box = slide3.shapes.add_textbox(x_left, y_pos, box_w, Inches(0.4))
    section_frame = section_box.text_frame
    section_frame.text = "Data Preparation"
    for paragraph in section_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
    
    y_pos += Inches(0.5)
    
    add_rounded_rectangle(slide3, x_left, y_pos, box_w, box_h,
                         "Collect Images", (176, 196, 222), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_left, y_pos, box_w, box_h,
                         "Manual Crop\n(manual_crop.py)", (176, 196, 222), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_left, y_pos, box_w, box_h,
                         "Label States", (176, 196, 222), (0, 0, 0))
    
    # Middle column - Training
    y_pos = Inches(1.3)
    x_mid = Inches(4)
    
    section_box = slide3.shapes.add_textbox(x_mid, y_pos, box_w, Inches(0.4))
    section_frame = section_box.text_frame
    section_frame.text = "Model Training"
    for paragraph in section_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 102, 0)
    
    y_pos += Inches(0.5)
    
    add_rounded_rectangle(slide3, x_mid, y_pos, box_w, box_h,
                         "Data Augmentation", (144, 238, 144), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_mid, y_pos, box_w, box_h,
                         "Train Classifier\n(train_classifier.py)", (144, 238, 144), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_mid, y_pos, box_w, box_h,
                         "Model Validation", (144, 238, 144), (0, 0, 0))
    
    # Right column - Deployment
    y_pos = Inches(1.3)
    x_right = Inches(7.2)
    
    section_box = slide3.shapes.add_textbox(x_right, y_pos, box_w, Inches(0.4))
    section_frame = section_box.text_frame
    section_frame.text = "Prediction"
    for paragraph in section_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(102, 0, 0)
    
    y_pos += Inches(0.5)
    
    add_rounded_rectangle(slide3, x_right, y_pos, box_w, box_h,
                         "Load Model", (255, 182, 193), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_right, y_pos, box_w, box_h,
                         "YOLO Detection", (255, 182, 193), (0, 0, 0))
    y_pos += spacing
    
    add_rounded_rectangle(slide3, x_right, y_pos, box_w, box_h,
                         "State Prediction\n(predict_veri.py)", (255, 182, 193), (0, 0, 0))
    
    # Evaluation box at bottom
    y_eval = Inches(5.5)
    x_eval = Inches(10.3)
    add_rounded_rectangle(slide3, x_eval, y_eval, box_w, box_h,
                         "Evaluation\n(evaluate_classifier.py)", (221, 160, 221), (0, 0, 0))
    
    # ==================== Slide 4: Key Features ====================
    slide4 = prs.slides.add_slide(title_slide_layout)
    
    # Background
    background = slide4.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(250, 250, 250)
    
    # Title
    title_box = slide4.shapes.add_textbox(
        Inches(0.5), Inches(0.3), Inches(12.33), Inches(0.6)
    )
    title_frame = title_box.text_frame
    title_frame.text = "Key Features & Capabilities"
    for paragraph in title_frame.paragraphs:
        paragraph.alignment = PP_ALIGN.CENTER
        for run in paragraph.runs:
            run.font.size = Pt(32)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
    
    # Features
    features = [
        ("🎯 Automatic Detection", "YOLO v8 for pan/pot localization"),
        ("🧠 Deep Learning", "ResNet50 transfer learning"),
        ("📊 High Accuracy", "100% on verification set"),
        ("🖼️ Visual Feedback", "Green wireframe marking"),
        ("⚡ Real-time Ready", "Optimized inference pipeline"),
        ("📈 Continuous Learning", "Easy retraining workflow")
    ]
    
    y_feature = Inches(1.5)
    x_left_col = Inches(1)
    x_right_col = Inches(7)
    feature_width = Inches(5)
    feature_height = Inches(0.9)
    feature_spacing = Inches(1.1)
    
    for i, (title, desc) in enumerate(features):
        x_pos = x_left_col if i < 3 else x_right_col
        y_pos = y_feature + (i % 3) * feature_spacing
        
        # Feature box
        box = add_rounded_rectangle(
            slide4, x_pos, y_pos, feature_width, feature_height,
            "", (230, 240, 255), (0, 0, 0)
        )
        
        # Title
        text_frame = box.text_frame
        text_frame.clear()
        p1 = text_frame.paragraphs[0]
        p1.text = title
        p1.alignment = PP_ALIGN.LEFT
        for run in p1.runs:
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        # Description
        p2 = text_frame.add_paragraph()
        p2.text = desc
        p2.alignment = PP_ALIGN.LEFT
        for run in p2.runs:
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(60, 60, 60)
    
    # Save presentation
    prs.save('Kitchen_Assist_Workflow.pptx')
    print("✓ PowerPoint presentation created: Kitchen_Assist_Workflow.pptx")


if __name__ == "__main__":
    create_workflow_presentation()
