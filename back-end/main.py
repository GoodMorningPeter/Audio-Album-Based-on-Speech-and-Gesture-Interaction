import os
import io
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request, render_template, Response
from flask_cors import CORS
from model import MobileNetV2
import pygame
from addtext import *
import cv2
import mediapipe as mp
from hand import*

app = Flask(__name__)
CORS(app)  # 解决跨域问题

curstate = "scenery"
curtext = "cat"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pygame.mixer.init()

def prepare_model(curstate):
    print(device)
    path_model = "./models/model_{}.pkl".format(curstate)   #调用训练好的模型
    model = torch.load(path_model)
    model = model.to(device)
    model.eval()
    return model

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "RGB":
        raise ValueError("input file does not RGB image...")
    return my_transforms(image).unsqueeze(0).to(device)


def get_prediction(image_bytes, model):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model(tensor).cpu()
    values, indices = outputs.data.max(1)

     # load class info
    class_json_path = "./static/json/class_indices_{}.json".format(curstate)
    assert os.path.exists(class_json_path), "class json path does not exist..."
    json_file = open(class_json_path, 'rb')

    class_indict = json.load(json_file)
    text= [class_indict[str(int(indices[0]))]]

    global curtext
    curtext = text[0]

    playsound()
    text = add_text(text)
    return_info = {"result": text}

    # playsound()

    return return_info

def playsound():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    pygame.mixer.music.load('./static/sound/{}.mp3'.format(curtext))  
    # pygame.mixer.music.load('./static/sound/lighting.mp3')  
    pygame.mixer.music.set_volume(0.5) 
    pygame.mixer.music.play()

def detect():
    global state, cur_state
    cnt = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        add_str = ""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame= cv2.flip(frame,1)
        results = hands.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_handedness:
            for hand_label in results.multi_handedness:
                hand_jugg=str(hand_label).split('"')[1]

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x*frame.shape[1]
                    y = hand_landmarks.landmark[i].y*frame.shape[0]
                    hand_local.append((x,y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    # print(gesture_str)
                    if gesture_str == "one":
                        if hand_jugg == "Left":
                            add_str = "next "
                        else:
                            add_str = "previous "
                        gesture_str = add_str + gesture_str

                    if gesture_str == "stop":
                        cur_state = 1
                    elif gesture_str == "go on":
                        cur_state = 2
                    elif gesture_str == "previous one":
                        cur_state = 3
                    elif gesture_str == "next one":
                        cur_state = 4
                    else:
                        cur_state = 0

                    if cur_state != state:
                        cnt += 1
                    if cnt >= 5:
                        state = cur_state
                        cnt = 0

                    cv2.putText(frame,gesture_str,(50,100),0,1.3,(255,255,0),2)

        if cv2.waitKey(1) & 0xFF == 27:
            break
            
        ret1,buffer = cv2.imencode('.jpg',frame)
        #将缓存里的流数据转成字节流
        frame = buffer.tobytes()
        #指定字节流类型image/jpeg
        yield  (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()
    cv2.destroyAllWindows()



model = prepare_model(curstate)

@app.route("/predict", methods=["POST"])
@torch.no_grad()
def predict():
    global curstate, model
    print(curstate)

    image = request.files["file"]
    img_bytes = image.read()
    info = get_prediction(image_bytes=img_bytes, model = model)
    return jsonify(info)

@app.route("/guide", methods=["POST"])
def guide():
    pygame.mixer.init()
    pygame.mixer.music.load('./static/sound/cat.mp3')  
    pygame.mixer.music.set_volume(0.5) 
    pygame.mixer.music.play()
    return jsonify("ok")

@app.route("/pause", methods=["POST"])
def pause():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.pause()
    return jsonify("ok")

@app.route("/cont", methods=["POST"])
def cont():
    pygame.mixer.music.unpause()
    return jsonify("ok")

@app.route("/again", methods=["POST"])
def again():
    # pygame.mixer.music.rewind()
    pygame.mixer.music.play()
    return jsonify("ok")

@app.route("/")
def index():
    # return render_template("index.html")
    return render_template("index.html")

@app.route("/scenery", methods=["GET", "POST"])
def scenery():
    global curstate, model
    curstate = "scenery"
    print(curstate) 
    model = prepare_model(curstate)
    return render_template("scenery.html")

@app.route("/animal", methods=["GET", "POST"])
def animal():
    global curstate, model
    curstate = "animal"
    print(curstate)
    model = prepare_model(curstate)
    return render_template("animal.html")

@app.route("/weather", methods=["GET", "POST"])
def weather():
    global curstate, model
    curstate = "weather"
    print(curstate)
    model = prepare_model(curstate)
    return render_template("weather.html")

@app.route("/gstate", methods=["POST"])
def gstate():
    global state
    print(state)
    if state == 1:
        pause()
    elif state == 2:
        cont()
    return str(state)


@app.route('/gdetect')
def gdetect():
    return Response(detect(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    cur_state = 0
    state = 0
    app.run(host="0.0.0.0", port=5000)
    






