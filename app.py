from PIL import Image, ImageDraw
import streamlit as st
import io
import os
import google.generativeai as genai
import cv2
import supervision as sv
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import time
from collections import defaultdict
from google.api_core.exceptions import ResourceExhausted
from transformers import CLIPProcessor, CLIPModel,BertTokenizer
import requests
from ultralytics import YOLO
from paddleocr import PaddleOCR
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Định nghĩa một hàm để cắt hình ảnh theo tọa độ
def crop_image(image, boxes, padding = 0):
    cropped_images = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox
        cropped_image = image.crop((x1- padding, y1- padding, x2+ padding, y2+ padding))
        cropped_images.append(cropped_image)
    return cropped_images   
def context(image, prompt):
    os.environ['GOOGLE_API_KEY'] = "AIzaSyDOX0ygiFy7AoUURHcesc5fRBPJQ8trRZ0"
    genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
    # Choose a Gemini model.
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([prompt, image])
    return response.text
def model_yoloworld(image):
  model = YOLO("yolov10x.pt")
  # Ensure image is in RGB mode
  #image = image[:,:,::-1]
  image = np.array(image)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  results = model(image, conf=0.2, classes= [0])  # only person
  #print(results)
  detections = sv.Detections.from_ultralytics(results[0])
  class_counts = defaultdict(int)
  labels = []
  if len(detections) > 0:
    for cls in detections["class_name"]:
        class_counts[cls] += 1
        labels.append(f"{cls}{class_counts[cls]}")
    detections["class_name"] = labels
  return detections

def get_prob_using_clip(image, list_of_class):
  model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  inputs = processor_clip(text=list_of_class, images=image, return_tensors="pt", padding=True)
  outputs =model_clip(**inputs)
  logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
  probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
  top_probs, top_indices = probs.topk(2, dim=1)
    # Filter probabilities greater than 0.5
  filtered_probs = top_probs[top_probs > 0.5]
  filtered_indices = top_indices[top_probs > 0.5]

  # If no probability is greater than 0.5, take the highest probability
  if len(filtered_probs) == 0:
      filtered_probs = top_probs[:, :1]  # take the highest probability
      filtered_indices = top_indices[:, :1]  # take the index of the highest probability
  # Map indices to class names
  filtered_classes = [ list_of_class[idx] for idx in filtered_indices.flatten().tolist()]

  return filtered_classes
def get_prob_using_clip_1(image, list_of_class):
  model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
  inputs = processor_clip(text=list_of_class, images=image, return_tensors="pt", padding=True)
  outputs =model_clip(**inputs)
  logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
  probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
  top_probs, top_indices = probs.topk(1, dim=1)
    # Filter probabilities greater than 0.5
  filtered_probs = top_probs[top_probs > 0.5]
  filtered_indices = top_indices[top_probs > 0.5]

  filtered_classes = [ list_of_class[idx] for idx in filtered_indices.flatten().tolist()]

  return filtered_classes

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(io.BytesIO(response.content))


# billboard

def overlap(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    return not (x1 > x2_ or x2 < x1_ or y1 > y2_ or y2 < y1_)


# Function to merge overlapping boxes
def merge_overlapping_boxes(boxes):
    if len(boxes) == 0:
        return []

    # Convert Ultralytics Boxes object to numpy array
    boxes_np = boxes.xyxy.cpu().numpy()

    # Your merging logic here (example)
    merged_boxes = []
    while len(boxes_np) > 0:
        current_box = boxes_np[0]
        boxes_np = boxes_np[1:]

        # Example of overlap function, adjust as needed
        def overlap(box1, box2):
            x1, y1, x2, y2 = box1[:4]
            x1_, y1_, x2_, y2_ = box2[:4]
            return not (x1 > x2_ or x2 < x1_ or y1 > y2_ or y2 < y1_)

        # Merge logic example, adjust based on your requirements
        overlapping_boxes = [current_box]
        for box in boxes_np:
            if overlap(current_box, box):
                overlapping_boxes.append(box)
            else:
                merged_boxes.append(merge(overlapping_boxes))
                current_box = box
                overlapping_boxes = [current_box]

        merged_boxes.append(merge(overlapping_boxes))

    return merged_boxes

def merge(boxes):
    # Implement your merging logic here
    # This is just an example, adjust as needed
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[2] for box in boxes)
    y2 = max(box[3] for box in boxes)
    return [x1, y1, x2, y2]

def levenshtein_distance(source, target):
    distance_table = [[0] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(len(source) + 1):
        distance_table[i][0] = i
    for j in range(len(target) + 1):
        distance_table[0][j] = j

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            if source[i - 1] == target[j - 1]:
                distance_table[i][j] = distance_table[i - 1][j - 1]
            else:
                insert = distance_table[i][j - 1] + 1
                delete = distance_table[i - 1][j] + 1
                substitute = distance_table[i - 1][j - 1] + 1
                distance_table[i][j] = min(insert, delete, substitute)

    return distance_table[len(source)][len(target)]

def get_key_with_min_value(dictionary):
    if not dictionary:
        return None, None

    min_value = min(dictionary.values())
    for key, value in dictionary.items():
        if value == min_value:
            return key, value

def detect_billboards(image):
    model = YOLO("billboard.pt")
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    # Perform prediction using the YOLO model
    val_results = model.predict(source=image, save=False, conf=0.25, show_labels=False)

    boxes = val_results[0].boxes.xyxy.cpu().numpy().tolist()

    # Merge overlapping bounding boxes
    merged_boxes = merge_overlapping_boxes(boxes)

    detected_brands = []
    list_brd = ["Larue", "Heineken", "Tiger", "Bivina", "Strongbow", "Edelweiss", "BiaViet", "LacViet", "SaiGon"]

    for idx, (x1, y1, x2, y2) in enumerate(merged_boxes):
        cropped_img = image[int(y1):int(y2), int(x1):int(x2)]

        result_ocr = ocr.ocr(cropped_img, cls=True)

        if not result_ocr or not result_ocr[0]:
            detected_brands.append("N/A")
            continue

        txts = [line[1][0] for line in result_ocr[0]]

        result_br = {}
        for b in list_brd:
            temp = [levenshtein_distance(txt.upper(), b.upper()) for txt in txts if len(txt) <= len(b)]
            if temp:
                result_br[b] = min(temp)

        detected_brand, value = get_key_with_min_value(result_br)
        if detected_brand and abs(len(detected_brand) - value) >= 3:
            detected_brands.append(detected_brand)
        else:
            detected_brands.append("N/A")

    labeled_boxes = [(x1, y1, x2, y2, detected_brands[i]) for i, (x1, y1, x2, y2) in enumerate(merged_boxes) if detected_brands[i] in list_brd]

    for (x1, y1, x2, y2, label) in labeled_boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = int((x1 + x2 - text_size[0]) / 2)
        text_y = int((y1 + y2 + text_size[1]) / 2)
        cv2.putText(image, label + " billboard", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return image, labeled_boxes


# Beer

def draw_bounding_boxes(image, boxes, names, label_colors):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        label = names[int(box.cls[0])]
        if label != "person":
            color = label_colors.get(label, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def check_and_filter_duplicates(boxes, names, existing_boxes, existing_names):
    filtered_boxes = []
    filtered_names = []
    for box, name in zip(boxes, names):
        if (box.xyxy[0], name) not in existing_boxes:
            filtered_boxes.append(box)
            filtered_names.append(name)
            existing_boxes.add((box.xyxy[0], name))
    return filtered_boxes, filtered_names

def process_uploaded_image(uploaded_image):
    model_path = 'beer.pt'
    label_colors = {
        'Bia Viet beer box': (255, 0, 0),  # Đỏ
        'Bivina-beer-box': (0, 128, 0),  # Xanh lá đậm
        'Edelweiss-beer-box': (0, 0, 255),  # Xanh dương
        'Heineken beer box': (0, 255, 255),  # Xanh ngọc
        'Laure beer box': (255, 0, 255),  # Tím
        'Strongbow-beer-box': (255, 255, 0),  # Vàng
        'Tiger beer box': (0, 0, 128),  # Xanh dương đậm
        # 'person': (255, 165, 0),  # Cam
    }

    model = YOLO(model_path)
    image = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    val_results = model.predict(source= image , save=False, conf=0.25, show_labels=False)
    boxes = val_results[0].boxes
    names = val_results[0].names
    output_image = draw_bounding_boxes(image, boxes, names , label_colors)
    print(names)
    label_count = defaultdict(int)
    for box in boxes:
            label = names[int(box.cls[0])]
            label_count[label] += 1
    return output_image,  label_count

    #return output_image, best_label_count


def activity(image_crop):
    activity = get_prob_using_clip(image_crop, ["Eating","Drinking","Smiling","Talking","Shopping", "Unknown"])
    return activity


def emotion_person(image_crop):
    see_face = get_prob_using_clip_1(image_crop, ["can see human face", "cannot see human face"])
    print(see_face)
    if see_face == []:
        emotion = ["Neural"]
    if "cannot see human face" in see_face:
        emotion = ["Neural"]
    else:
        emotion = get_prob_using_clip(image_crop, ["Happy", "Angry", "Enjoyable", "Relaxed", "Neutral", "Unknown"])
        return emotion

def type_of_people(image_crop):
    type_of = get_prob_using_clip(image_crop, ["Drinker",  "Buyer/Customer", "Unknown"])
    return type_of




if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

tabs = st.sidebar.radio("Navigation", ('Home', 'Results'))

if tabs == 'Home':
    st.title('Welcome to ĐPTV team')
    st.write('A topic on beer recognition for the AI Heineken competition could involve developing a system to identify and classify Heineken beer products using image or video recognition technology. This practical application could be applied to quality control processes, inventory management, and production monitoring to ensure accuracy and efficiency in manufacturing and distribution management.')

elif tabs == 'Results':
    st.title('Results')
    uploaded_file = st.sidebar.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        uploaded_image = Image.open(io.BytesIO(bytes_data))

        st.session_state.uploaded_image = Image.open(io.BytesIO(bytes_data))
        processed_image, label_count = process_uploaded_image(uploaded_image)

    if st.session_state.uploaded_image is not None:
        st.image( st.session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)

        # About person
        if st.checkbox("About person"):
            detections = model_yoloworld(st.session_state.uploaded_image)
            num_of_people = len(detections)
            st.write(f'Number of people appearing in the image: {num_of_people}')

            
            probabilities = get_prob_using_clip(st.session_state.uploaded_image, ["indoor", "outdoor"])

            # Generate HTML for labels and probabilities on the same row
            combined_html = "".join([
                f"<div style='display: inline-block; border: color: white; padding: 5px; margin: 5px; border-radius: 5px; text-align: center; min-width: 80px;'><strong>{'Environment'}</strong></div>"
                f"<div style='display: inline-block; color: white; background-color: green; padding: 5px; margin: 5px; border-radius: 5px; text-align: center; min-width: 80px;'>{prob}</div>"
                for prob in (probabilities)
            ])

            st.markdown(combined_html, unsafe_allow_html=True)

            scene = get_prob_using_clip(st.session_state.uploaded_image, ["bar", "pub","restaurant", "grocery store", "supermarket", "market", "shop","party", "celebration", "gathering", "happy hour", "fun time"])
            combined_html = "".join([
                f"<div style='display: inline-block; border: color: white; padding: 5px; margin: 5px; border-radius: 5px; text-align: center; min-width: 80px;'><strong>{'Scene'}:</strong></div>"
                f"<div style='display: inline-block; color: white; background-color: green; padding: 5px; margin: 5px; border-radius: 5px; text-align: center; min-width: 80px;'>{prob}</div>"
                for prob in (scene)
            ])
            
            st.markdown(combined_html, unsafe_allow_html=True)
        
            annotated_image = np.array(st.session_state.uploaded_image)
            annotated_image  = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
            LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1.5, text_color=sv.Color.BLACK)

            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image  = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            annotated_image_pil = Image.fromarray(annotated_image)
            #output = context(annotated_image_pil, prompt)
            #st.write(output)
            st.image(annotated_image, caption='Annotated Image', use_column_width=False)

            cropped_images = crop_image(st.session_state.uploaded_image, detections.xyxy)
            cropped_images_padding = crop_image(st.session_state.uploaded_image, detections.xyxy, padding=40)

            max_images_per_row = 4
            count = 0
            num_rows = (len(cropped_images) + max_images_per_row - 1) // max_images_per_row
     
            for row in range(num_rows):
                    row_images = cropped_images[row * max_images_per_row:(row + 1) * max_images_per_row]
                    row_images_padding = cropped_images_padding[row * max_images_per_row:(row + 1) * max_images_per_row]
                    cols = st.columns(len(row_images))  
                    for idx, cropped_image in enumerate(row_images):
                        with cols[idx]:
                            label = detections['class_name'][count]
                            activities = activity(cropped_image)
                            emotion_person_1 = emotion_person(cropped_image)
                            type_of_people_1 = type_of_people(cropped_image)
                            
                            

                            caption = f"{label}: \n {activities} {emotion_person_1 } {type_of_people_1}" #####
                            count = count + 1
                            st.image(cropped_image, caption=caption)


        if st.checkbox("About beer"):
            if st.session_state.uploaded_image is not None:
                processed_image, label_count = process_uploaded_image(st.session_state.uploaded_image)

                st.write("Label Counts:")
                combined_html = ""
                for label, count in label_count.items():
                    if label != "person":
                        combined_html += f"""
                            <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                                <div style='flex-grow: 1; font-weight: bold;'>{label}:</div>
                                <div style='flex-shrink: 0; background-color: #4CAF50; color: white; padding: 5px 10px; border-radius: 5px;'>{count}</div>
                            </div>
                        """
                st.markdown(combined_html, unsafe_allow_html=True)

                # Display annotated image with bounding boxes
                annotated_image = processed_image.copy()
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image, caption='Annotated Image', use_column_width=False)


        if st.checkbox("About billboard"):
            if st.session_state.uploaded_image is not None:
                processed_image, label_count = process_uploaded_image(st.session_state.uploaded_image)

                st.write("Label Counts:")
                for label, count in label_count.items():
                    if label != "person":
                        st.write(f"{label}: {count}")

                # Perform billboard detection
                model = YOLO("billboard.pt")
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                val_results = model.predict(source=np.array(st.session_state.uploaded_image), save=False, conf=0.25, show_labels=False)
                boxes = val_results[0].boxes
                names = val_results[0].names

                # Merge overlapping boxes
                merged_boxes = merge_overlapping_boxes(boxes)

                detected_brands = []
                list_of_brands = ["Larue", "Heineken", "Tiger", "Bivina", "Strongbow", "Edelweiss", "BiaViet", "LacViet", "SaiGon"]

                for idx, (x1, y1, x2, y2) in enumerate(merged_boxes):
                    cropped_img = st.session_state.uploaded_image.crop((x1, y1, x2, y2))
                    cropped_img_np = np.array(cropped_img)

                    result_ocr = ocr.ocr(cropped_img_np, cls=True)

                    if not result_ocr or not result_ocr[0]:
                        detected_brands.append("N/A")
                        continue

                    txts = [line[1][0] for line in result_ocr[0]]

                    result_brand = {}
                    for brand in list_of_brands:
                        temp = [levenshtein_distance(txt.upper(), brand.upper()) for txt in txts if len(txt) <= len(brand)]
                        if temp:
                            result_brand[brand] = min(temp)

                    detected_brand, value = get_key_with_min_value(result_brand)
                    if detected_brand and abs(len(detected_brand) - value) >= 3:
                        detected_brands.append(detected_brand)
                    else:
                        detected_brands.append("N/A")

                labeled_boxes = [(x1, y1, x2, y2, detected_brands[i]) for i, (x1, y1, x2, y2) in enumerate(merged_boxes) if detected_brands[i] in list_of_brands]
                #processed_image = np.array(processed_image)
                processed_image = Image.fromarray(processed_image)
                for (x1, y1, x2, y2, label) in labeled_boxes:
                    draw = ImageDraw.Draw(processed_image)
                    draw.rectangle([(x1, y1), (x2, y2)], outline="blue", width=2)
                    draw.text((x1, y1), f"{label} billboard", fill="red")

                processed_image = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
                st.image(processed_image, caption='Annotated Image')






