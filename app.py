import streamlit as st
import os
import time
import numpy as np
import glob as glob
# from sklearn.model_selection import train_test_split
# from keras.utils import np_utils
import librosa as lb
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import cv2
import streamlit.components.v1 as stc
from PIL import Image

def voice_input():
   
    duration = 3  # 3秒間録音する
    # i = i+100

    # デバイス情報関連
    # sd.default.device = [1, 3] # Input, Outputデバイス指定
    input_device_info = sd.query_devices(device=sd.default.device[1])
    sr_in = int(input_device_info["default_samplerate"])

    # st.write("Wait...")
    # time.sleep(3)

    # # for i in range(3):
    # #     st.subheader(f'{i}')
    # #     time.sleep(1)

    # st.subheader("start")
    # time.sleep(1)

    
    # 録音
    myrecording = sd.rec(int(duration * sr_in), samplerate=sr_in, channels=2)
    sd.wait() # 録音終了待ち

    print(myrecording.shape) #=> (duration * sr_in, channels)

    # 録音信号のNumPy配列をwav形式で保存
    sf.write("myrecording1000.wav", myrecording, sr_in)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'secret.json'

def png_trans():
    y,sr = lb.load(f'myrecording1000.wav') # 音声読み込み
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, win_length=512, hop_length=512) # メルスペクトログラムに変換
    S_dB = librosa.power_to_db(S, ref=np.max)

    # プロット
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='linear') #yの値はlinearにしないとおかしくなった
    # filename = filename.replace('.wav','')
    # filename = filename.replace('C:/Users/20t311/Documents/大学関連/実験2/myvoice/angry/','')
    # filename = f'{filename}'

    plt.savefig("myrecording1000.png", format="png", dpi=300)

def speech_trans():
    r = sr.Recognizer()

    audio_file = "myrecording1000.wav"
    print(audio_file)

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        print(audio)

    try:
        # print(r.recognize_google(audio, language='ja-JP'))
        text = r.recognize_google(audio, language='ja-JP')
        return text
        # text = "a"
        # print(text)
    except sr.UnknownValueError:
        text = "error（聞き取れなかったよごめんね）"
        # print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        # print("Could not request results from Google Speech Recognition service; {0}".format(e))
        text = "error"
    return text

def voice_color_see_test(model):
    voice_input()
    png_trans()
    text = speech_trans()

    img = cv2.imread("myrecording1000.png")
    # cv2.imshow('camera' , frame)
    img = cv2.resize(img,dsize=(64,64))
    img = img.astype('float32')
    img /= 255.0
    img = img[None, ...]

    int2emotion = {
    "01": "neutral",
    "02": "expectation",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised"
}
    # 予測 --- (※8)
    res = model.predict([img])[0]
    v = res.argmax()
    v2 = 0
    if v == 0:
      color = 'neutral'
    elif v == 1:
      color = 'expection'
    elif v == 2:
      print(v)
    #   res2 = hpan_model.predict([img])[0]
      color = 'happy'
      
    elif v == 3:
      color = 'sad'
    elif v == 4:
    #   res2 = hpan_model.predict([img])[0]
    #   v2 = res2.argmax()
    #   if v2 == 0:
    #     v = 2
    #     color = 'happy'
    #   else:
    #     v = 4
      color = 'angry'
    elif v == 5:
      color = 'fear'
    elif v == 6:
      color = 'disgust'
    elif v == 7:
      color = 'surprised'
    print("------------------------------------------------")
    # print(v2)
    print(v)
    print(color)
    # print(type(text))
    # if v == 0:
    #     st.write(f'<span style="color:#0F0">{text}</span>', unsafe_allow_html=True)
    # elif v == 1:
    #     st.write(f'<span style="color:orange">{text}</span>', unsafe_allow_html=True)
    # elif v == 2:
    #     st.write(f'<span style="color:yellow">{text}</span>', unsafe_allow_html=True)
    # elif v == 3:
    #     st.write(f'<span style="color:blue">{text}</span>', unsafe_allow_html=True)
    # elif v == 4:
    #     st.write(f'<span style="color:red">{text}</span>', unsafe_allow_html=True)
    # elif v == 5:
    #     st.write(f'<span style="color:green">{text}</span>', unsafe_allow_html=True)
    # elif v == 6:
    #     st.write(f'<span style="color:purple">{text}</span>', unsafe_allow_html=True)
    # elif v == 7:
    #     st.write(f'<span style="color:0FF">{text}</span>', unsafe_allow_html=True)
        
        
    return text,v

    # stc.html("<p style='color:red;'> Streamlit is Awesome")


st.set_page_config(
    page_title="VoiSee", 
    # page_icon=image, 
    layout="centered", 
    # initial_sidebar_state="auto", 
)

model = load_model('cnn_200.h5')
if 'count' not in st.session_state: 
	st.session_state.count = 0 #countがsession_stateに追加されていない場合，0で初期化

st.title('感情可視化アプリVoiSee')
# st.header('概要')
st.markdown('#### 貴方の感情に合わせて文字の色が変わる文字起こしアプリです')
st.write('3秒間話してください')


if st.button('開始'):
    comment = st.empty()
    comment.caption('文字起こしを開始します')
    a,emo = voice_color_see_test(model)
    comment.caption('完了しました')
    st.session_state[f'{st.session_state.count}'] = a
    st.session_state[f'emotion{st.session_state.count}'] = emo

    st.session_state.count += 1

# if st.button('履歴を表示'):
for i in range(st.session_state.count)[::-1]:
    text = st.session_state[f'{i}']
    x = st.session_state[f'emotion{i}']
    col1, col2 = st.columns(2)
    # st.write(f'<span style="color:red">{text}</span>', unsafe_allow_html=True)
    if x == 0:
        with col1:
            img = Image.open("emokao_01.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='普通（信頼）', use_column_width='never')
        with col2:
            st.write(f'<span style="color:#0F0;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 1:
        with col1:
            img = Image.open("emokao-02.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='期待', use_column_width='never')
        with col2:
            st.write(f'<span style="color:orange;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 2:
        with col1:
            img = Image.open("emokao-03.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='喜び', use_column_width='never')
        with col2:
            st.write(f'<span style="color:#f7ce00;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 3:
        with col1:
            img = Image.open("emokao-04.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='悲しみ', use_column_width='never')
        with col2:
            st.write(f'<span style="color:blue;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 4:
        with col1:
            img = Image.open("emokao-05.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='怒り', use_column_width='never')
        with col2:
            st.write(f'<span style="color:red;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 5:
        with col1:
            img = Image.open("emokao-07.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='恐怖', use_column_width='never')
        with col2:
            st.write(f'<span style="color:green;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
    elif x == 6:
        with col1:
            img = Image.open("emokao-06.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='嫌悪', use_column_width='never')
            # stc.html("<img src='C:/Users/20t311/Documents/大学関連/実験2/code/emokao-07.jpg'>")
        with col2:
            st.write(f'<span style="color:purple;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
    elif x == 7:
        with col1:
            img = Image.open("emokao-08.jpg")
            # 読み込んだ画像の幅、高さを取得し半分に
            (width, height) = (img.width // 2, img.height // 2)
            # 画像をリサイズする
            img_resized = img.resize((width, height))
            st.image(img_resized,caption='驚き', use_column_width='never')
        with col2:
            st.write(f'<span style="color:#0FF;font-size:20pt">{text}</span>', unsafe_allow_html=True)
        st.write("---------------------------------------------------------------------------------------")
        
