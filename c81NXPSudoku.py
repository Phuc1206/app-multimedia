# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 09:25:03 2024

@author: Xuanp
"""
##########################
# B1: NẠP THƯ VIỆN
########################
 # cv: thị giác máy tính Computer  Vison = HÌNH ẢNH & VIDEO CLIP
import cv2 #THƯ VIỆN COMPUTER VISION Version 2 

 #speech
import speech_recognition as sr
from gtts import gTTS
import pygame
from pydub import AudioSegment
from googletrans import Translator
 #GUI
import tkinter as tk 
#from tkinter import ttk 
from tkinter import messagebox,filedialog, simpledialog
import numpy as np
#Video
from PIL import Image, ImageTk 
#Sudoku
from c83NXPMainSudoku import run_real_time_sudoku
 #các thư viện khác
#import numpy as np  #THƯ VIỆN Mummeric Python = LẬP MA TRẬN GIỮ CHỖ ĐỀ GHÉP 2 KHUNG HÌNH (Color và hình lệch Gray)   
import os  # THƯ VIỆN OS MS. WINDOWS = Lập thư mục & lưu file khung ảnh (frame)
import time

###################################
# B2: KHAI BÁO CÁC BIẾN TOÀN CỤC 
##################################
# Âm thanh
afile = 'C08NXP.mp3' #  mẫu tên  file âm thanh .mp3 Input sẽ lưu 
adir = 'C081NXuanPhucA'     # Thư mục lưu các file [trên] 

# Hình ảnh
pfile = 'download.jpg'
pdir = 'C082NXuanPhucP'

# Video
vfile = 'video.mp4'
vdir = 'C083NXuanPhucV'
vdirf = 'C084NXuanPhucVF'
vdirfacer = 'C085NXuanPhucVFace'

#Text
tfile = "ketqua.txt"
tdir = 'C086NXuanPhucT'

# khác
count = 0
pygame.mixer.init()

r = sr.Recognizer() 
###################################
# B3: THỰC HIỆN CÁC THỦ TỤC CƠ BẢN : HỆ THỐNG,..
##################################

"""
TẠO THƯ MỤC
"""
os.makedirs(adir, exist_ok=True) # TẠO THƯ MỤC LƯU (từ thư viện os - của OS MS. Windows) = mất được = v
#Không lập thư mục hình ảnh pdir và video vdir = vì bên trong các thư mục này đã có sẵn các file media thực nghiệm[tránh mất file]
# hàm thoát 
def Thoat(event=None): 
 traloi = messagebox.askquestion("Xác nhận", "Bạn có muốn thoát không (Y/N)?")
 if traloi == "yes":  
     wn.destroy()  
     cv2.destroyAllWindows()
    
# hàm nhận lệnh = âm thanh (speech)

###################################
# B4: THỰC HIỆN CÁC THỦ TỤC XỬ LÝ
#######################################
#PHẦN 1: VOICE = SPEECH = AUDIO = SOUND
########################################
#4.1 INPUT =  Ra lệnh bằng voice (speech)
def Lenh():  # NHẬP ÂM THANH TỪ MICROPHONE
    global audio_data, afile
    a = txtSourceR.get()
    aS = txtSourceS.get()
    
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
        
    with sr.Microphone() as Source:
        # Hiệu chỉnh MIC trước khi ra lệnh
        messagebox.showinfo("Nhắc nhở", "Hiệu chỉnh MIC trước khi ra lệnh bằng lời nói! " + format(a) + " giây")
        r.adjust_for_ambient_noise(Source, duration=int(a))
        
        # Thông báo và bắt đầu nhận lệnh
        messagebox.showinfo("Cảnh báo", "Ra lệnh bằng tiếng Việt trong " + format(aS) + " giây, bấm OK để bắt đầu đọc lệnh.")
        audio_data = r.record(Source, duration=int(aS))
        
        try:
            # Nhận diện giọng nói
            tt = r.recognize_google(audio_data, language="vi")
        except:
            tt = "Quý vị nói gì nghe không rõ...!"
        
        # Hiển thị nội dung đã nhận diện
        messagebox.showinfo("Quý vị đã nói là:", format(tt))
        
        # Tạo tệp âm thanh từ văn bản
        vx = gTTS(text=tt, lang='vi')
        c = time.localtime()
        OUT_FILE = "NXP%d.mp3" % int(time.mktime(c))
        vx.save(os.path.join(adir, OUT_FILE))
        messagebox.showinfo("Thông báo", f"Nội dung đã được lưu vào tệp mới: {OUT_FILE}")
        
def chooseAudio():
    global afile
    afile = filedialog.askopenfilename(
        title="Chọn tệp âm thanh", 
        filetypes=(("Audio Files", "*.mp3;*.wav"), ("All Files", "*.*"))
    )
    
    if afile:
        messagebox.showinfo("Thông báo", f"Đã chọn âm thanh: {os.path.basename(afile)}")
        lblfileA.configure(text = "File Audio: %s" %afile)
    else:
        messagebox.showwarning("Cảnh báo", "Chưa chọn âm thanh!")
        
#4.2 OUTPUT = XUẤT RA LỜI NÓI THEO VĂN BẢN ĐÃ NHẬP = Trả lời bằng tiếng Việt : Text => Nói tiếng Việt

def Doc():
    global afile
    default_file = os.path.abspath(os.path.join(adir, afile))
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    # Kiểm tra tệp
    if os.path.exists(afile):
        pygame.mixer.music.load(afile)
    elif os.path.exists(default_file):
        pygame.mixer.music.load(default_file)
    else:
        messagebox.showerror("Lỗi", "Không tìm thấy tệp âm thanh!")
        return
    
    # Phát âm thanh
    pygame.mixer.music.play()
    messagebox.showinfo("Thông báo", "Đang phát âm thanh.")
def translate_audio_with_playback():
    global afile
    default_file = os.path.abspath(os.path.join(adir, afile))
    
    if not os.path.exists(default_file):
        messagebox.showerror("Lỗi", "Không tìm thấy tệp âm thanh!")
        return
    
    # Chuyển đổi MP3 sang WAV nếu cần
    wav_file = os.path.splitext(default_file)[0] + '.wav'
    if not os.path.exists(wav_file):  # Chỉ chuyển đổi nếu file WAV chưa tồn tại
        try:
            audio = AudioSegment.from_mp3(default_file)
            audio.export(wav_file, format="wav")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chuyển đổi tệp MP3 sang WAV: {e}")
            return
    
    try:
        with sr.AudioFile(wav_file) as source:
            r = sr.Recognizer()
            audio_data = r.record(source)
            
            # Kiểm tra nếu dữ liệu âm thanh là rỗng
            if len(audio_data.frame_data) == 0:
                messagebox.showerror("Lỗi", "Không có dữ liệu âm thanh để nhận diện!")
                return
            
            # Nhận diện văn bản từ âm thanh
            recognized_text = r.recognize_google(audio_data, language="vi")
            
            # Kiểm tra nếu nhận diện không thành công
            if recognized_text is None or recognized_text == "":
                messagebox.showerror("Lỗi", "Không thể nhận diện âm thanh!")
                return
            
            messagebox.showinfo("Nhận diện", f"Nội dung nhận diện: {recognized_text}")
        
        # Dịch văn bản nhận diện
        translated = Translator().translate(recognized_text, src="vi", dest="en")
        messagebox.showinfo("Dịch", f"Nội dung dịch: {translated.text}")
        
        # Phát đoạn dịch
        play_translation(translated.text)
    
    except sr.UnknownValueError:
        messagebox.showerror("Lỗi", "Google Speech Recognition không thể nhận diện được âm thanh!")
    except sr.RequestError as e:
        messagebox.showerror("Lỗi", f"Không thể kết nối đến dịch vụ Google Speech Recognition: {e}")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể nhận diện hoặc dịch đoạn âm thanh: {e}")

def play_translation(text):
    try:
        tts = gTTS(text=text, lang='en')
        c = time.localtime()
        OUT_FILE = "NXPtranslated_audio%d.mp3" % int(time.mktime(c))
        tts.save(os.path.join(adir, OUT_FILE))
        
        
        # Phát âm thanh
        pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(adir, OUT_FILE))
        pygame.mixer.music.play()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể phát âm đoạn dịch: {e}")



 #####################################################
 #PHẦN 2: CÁC THỦ TỤC XỬ LÝ IMAGE = PICTURE = ....
 #####################################################
def choose_image():
    global pfile, pdir
    pfile = filedialog.askopenfilename(
        title="Chọn tệp ảnh",
        filetypes=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"), ("All Files", "*.*"))
    )
    if pfile:
        pdir = os.path.dirname(pfile) 
        messagebox.showinfo("Thông báo", f"Đã chọn ảnh: {os.path.basename(pfile)}")
        lblfileP.configure(text = "File Audio: %s" %pfile)
    else:
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
 # CHUYỂN ẢNH =>ẢNH XÁM img
def pGray():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return  # Thoát hàm nếu không có ảnh được chọn

    img = cv2.imread(os.path.join(pdir, pfile), cv2.IMREAD_GRAYSCALE)
    if img is None:  # Kiểm tra nếu ảnh không hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return

    cv2.imshow('Anh Xam', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Lưu ảnh xám
    filename, ext = os.path.splitext(pfile)  # Lấy tên tệp và phần mở rộng
    save_path = os.path.join(pdir, f"{filename}_gray{ext}")  # Tạo tên tệp mới với '_gray'
   
    # Lưu ảnh xám vào tệp
    cv2.imwrite(save_path, img)
    messagebox.showinfo("Thông báo", f"Ảnh xám đã được lưu tại {save_path}")

 # LẤY SIZE ẢNH XÁM (2 D)
def pGraySize(img):
  (h, w) = img.shape
  messagebox.showinfo("width={}, height={}".format(w, h))

# ĐỌC ẢNH MÀU
def pColor():
 img = cv2.imread(os.path.join(pdir, pfile)) #đọc ảnh gốc
 cv2.imshow('Anh Mau',img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

 # LẤY SIZE ẢNH MÀU (3 D) img2
def pColorSize():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return
    img = cv2.imread(os.path.join(pdir, pfile))  # Đọc ảnh từ file
    if img is None:  # Kiểm tra ảnh hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return
    (h, w, d) = img.shape
    messagebox.showinfo("Thông tin size ảnh","width={}, height={}, depth = {}".format(w, h, d))

 # LẤY GIÁ TRỊ MÀU CỦA ĐIỂM ẢNH (Pixel) với hệ màu RGB
def getRGBPixel():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return
    
    img = cv2.imread(os.path.join(pdir, pfile))  # Đọc ảnh từ file
    if img is None:  # Kiểm tra ảnh hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return
    
    # Hiển thị cửa sổ nhập tọa độ
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter

    try:
        # Yêu cầu người dùng nhập tọa độ x và y
        x = int(simpledialog.askstring("Nhập tọa độ", "Nhập x (tọa độ ngang):", parent=root))
        y = int(simpledialog.askstring("Nhập tọa độ", "Nhập y (tọa độ dọc):", parent=root))
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập giá trị hợp lệ cho tọa độ!")
        root.destroy()
        return
    
    # Kiểm tra xem tọa độ có hợp lệ không
    if x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]:
        messagebox.showwarning("Cảnh báo", "Tọa độ ngoài phạm vi ảnh!")
        root.destroy()
        return
    
    # Lấy giá trị màu của pixel tại tọa độ (x, y)
    (B, G, R) = img[y, x]  # Lưu ý là tọa độ y (dọc) và x (ngang)
    
    # Hiển thị thông tin màu sắc của pixel
    messagebox.showinfo("Thông tin màu ảnh", "Red={}, Green={}, Blue={}".format(R, G, B))
    root.destroy()

 # CẮT ẢNH 
def pColorPart():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return
    
    img = cv2.imread(os.path.join(pdir, pfile))  # Đọc ảnh từ file
    if img is None:  # Kiểm tra ảnh hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return
    
    h, w, d = img.shape  # Lấy chiều cao, chiều rộng và độ sâu của ảnh
    root = tk.Tk()
    root.withdraw()
    try:
        x1 = int(simpledialog.askstring("Nhập tọa độ", "Nhập x1 (tọa độ bắt đầu của cắt ngang):", parent=root))
        y1 = int(simpledialog.askstring("Nhập tọa độ", "Nhập y1 (tọa độ bắt đầu của cắt dọc):", parent=root))
        x2 = int(simpledialog.askstring("Nhập tọa độ", "Nhập x2 (tọa độ kết thúc của cắt ngang):", parent=root))
        y2 = int(simpledialog.askstring("Nhập tọa độ", "Nhập y2 (tọa độ kết thúc của cắt dọc):", parent=root))
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập giá trị hợp lệ cho tọa độ!")
        root.destroy()
        return
    # Kiểm tra tính hợp lệ của các tọa độ
    if x1 >= x2 or y1 >= y2:
        messagebox.showwarning("Cảnh báo", "Tọa độ cắt không hợp lệ!")
        root.destroy()
        return
    
    # Kiểm tra xem các tọa độ có vượt quá kích thước của ảnh không
    if x2 > w or y2 > h:
        messagebox.showwarning("Cảnh báo", "Tọa độ cắt vượt quá kích thước ảnh!")
        root.destroy()
        return
    
    # Cắt ảnh theo tọa độ đã cho
    part_img = img[y1:y2, x1:x2]  # Cắt ảnh từ (x1, y1) đến (x2, y2)
    root.destroy()
    # Hiển thị phần ảnh cắt
    cv2.imshow('Phan anh tu {}-{} x {}-{}'.format(x1, y1, x2, y2), part_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Tạo tên mới để lưu ảnh
    filename, ext = os.path.splitext(pfile)  # Lấy tên tệp và phần mở rộng
    save_path = os.path.join(pdir, f"{filename}_cut_{x1}_{y1}_{x2}_{y2}{ext}")  # Tạo tên tệp mới với tọa độ cắt
    
    # Lưu ảnh đã cắt
    cv2.imwrite(save_path, part_img)  # Lưu ảnh đã cắt vào tệp
    messagebox.showinfo("Thông báo", f"Ảnh đã được lưu tại {save_path}")
 #ROTATE
def pRotate():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return
    
    img = cv2.imread(os.path.join(pdir, pfile))  # Đọc ảnh từ file
    if img is None:  # Kiểm tra ảnh hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return
    
    h, w, d = img.shape  # Lấy chiều cao, chiều rộng và độ sâu của ảnh
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    
    try:
        a = float(simpledialog.askstring("Nhập góc quay", "Nhập góc quay (độ):", parent=root))
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập giá trị hợp lệ cho góc quay!")
        root.destroy()
        return

    # Kiểm tra xem giá trị góc có hợp lệ không
    if a < -360 or a > 360:
        messagebox.showwarning("Cảnh báo", "Góc quay không hợp lệ! Vui lòng nhập góc quay trong khoảng từ -360 đến 360 độ.")
        root.destroy()
        return
    
    # Tính toán và thực hiện quay ảnh
    center = (w // 2, h // 2)  # Tính trung tâm ảnh
    M = cv2.getRotationMatrix2D(center, a, 1.0)  # Lấy ma trận quay
    rotated = cv2.warpAffine(img, M, (w, h))  # Áp dụng ma trận quay lên ảnh
    root.destroy()
    # Hiển thị ảnh sau khi quay
    cv2.imshow(f'Anh sau quay {a} do', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Tạo tên mới để lưu ảnh
    filename, ext = os.path.splitext(pfile)  # Lấy tên tệp và phần mở rộng
    save_path = os.path.join(pdir, f"{filename}_rotated_{int(a)}{ext}")  # Tạo tên tệp mới với góc quay
    
    # Lưu ảnh đã quay
    cv2.imwrite(save_path, rotated)  # Lưu ảnh đã quay vào tệp
    messagebox.showinfo("Thông báo", f"Ảnh đã được lưu tại {save_path}")


 # RESIZE
def pResize():
    if not pfile:  # Kiểm tra xem người dùng có chọn ảnh chưa
        messagebox.showwarning("Cảnh báo", "Chưa chọn ảnh!")
        return
    
    img = cv2.imread(os.path.join(pdir, pfile))  # Đọc ảnh từ file
    if img is None:  # Kiểm tra ảnh hợp lệ
        messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
        return
    
    h, w, d = img.shape  # Lấy chiều cao, chiều rộng và độ sâu của ảnh
    root = tk.Tk()
    root.withdraw()  # Ẩn cửa sổ chính của Tkinter
    
    try:
        z = float(simpledialog.askstring("Nhập tỷ lệ co giãn", "Nhập tỷ lệ co giãn (%):", parent=root))
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập giá trị hợp lệ cho tỷ lệ co giãn!")
        root.destroy()
        return

    # Kiểm tra xem tỷ lệ có hợp lệ không
    if z <= 0:
        messagebox.showwarning("Cảnh báo", "Tỷ lệ co giãn không hợp lệ! Vui lòng nhập tỷ lệ lớn hơn 0.")
        root.destroy()
        return
    
    # Tính toán tỷ lệ co giãn và thay đổi kích thước ảnh
    r = z / 100  # Chuyển đổi từ phần trăm sang tỷ lệ
    dim = (int(w * r), int(h * r))  # Tính kích thước mới
    resized_img = cv2.resize(img, dim)  # Thay đổi kích thước ảnh
    root.destroy()
    # Hiển thị ảnh sau khi resize
    cv2.imshow(f'Anh sau Resize {z}%', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Tạo tên mới để lưu ảnh
    filename, ext = os.path.splitext(pfile)  # Lấy tên tệp và phần mở rộng
    save_path = os.path.join(pdir, f"{filename}_resized_{int(z)}{ext}")  # Tạo tên tệp mới với tỷ lệ co giãn
    
    # Lưu ảnh đã resize
    cv2.imwrite(save_path, resized_img)  # Lưu ảnh đã resize vào tệp
    messagebox.showinfo("Thông báo", f"Ảnh đã được lưu tại {save_path}")
# HÀM GHÉP Color Frame với Gray Frame
def print_image(img, diff_im):
    new_img = np.zeros([img.shape[0], img.shape[1] * 2, img.shape[2]])
    new_img[:, :img.shape[1], :] = img
    new_img[:, img.shape[1]:, 0] = diff_im
    new_img[:, img.shape[1]:, 1] = diff_im
    new_img[:, img.shape[1]:, 2] = diff_im
    cv2.imshow('diff', new_img)
    return new_img
def choose_video():
    global vfile
    vfile = filedialog.askopenfilename(
        title="Chọn tệp video",
        filetypes=(("Video Files", "*.mp4;*.avi;*.mov"), ("All Files", "*.*"))
    )
    if vfile:
        messagebox.showinfo("Thông báo", f"Đã chọn video: {os.path.basename(vfile)}")
        lblfileV.configure(text = "File Video: %s" %vfile)
    else:
        messagebox.showwarning("Cảnh báo", "Chưa chọn video!")
# Chức năng cắt video thành các frame
def cut_video_frames():
    global vfile
    if not vfile:
        messagebox.showinfo("Thông báo", f"Đã chọn video mặc định: {os.path.basename(vfile)}")
    if not os.path.exists(os.path.join(vdir, vfile)):
        messagebox.showerror("Lỗi", f"Không tìm thấy video: {vfile}")
        return

    cap = cv2.VideoCapture(os.path.join(vdir, vfile))
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở video!")
        return
    
    save_dir = vdirf
    os.makedirs(save_dir, exist_ok=True)
    
    idx = -1
    last_gray = None

    while True:
        ret, frame = cap.read()
        idx += 1
        if not ret:
            messagebox.showinfo("Thông báo", f"Đã cắt xong video: {os.path.basename(vfile)}")
            break
        
        # Chuyển khung hình hiện tại sang ảnh xám
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Nếu là khung hình đầu tiên, bỏ qua so sánh khác biệt
        if last_gray is None:
            last_gray = gray
            continue
        
        # Tính sự khác biệt giữa khung hình hiện tại và trước đó
        diff = cv2.absdiff(gray, last_gray)
        new_img = print_image(frame, diff)
        
        # Lưu frame đã ghép vào thư mục
        cv2.imwrite(os.path.join(save_dir, f'frame_{idx:06d}.jpg'), new_img)
        last_gray = gray

        print(f"Lưu frame thứ {idx}...")

    cap.release()
    cv2.destroyAllWindows()
def VideoFrame():
    global vfile
    if not vfile:
        messagebox.showinfo("Thông báo", f"Đã chọn video mặc định: {os.path.basename(vfile)}")
    if not os.path.exists(os.path.join(vdir, vfile)):
        messagebox.showerror("Lỗi", f"Không tìm thấy video: {vfile}")
        return
    cap = cv2.VideoCapture(os.path.join(vdir, vfile))
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    c = 0  # Bộ đếm số khung hình
    while True:
        ret, frame = cap.read()  # Đọc khung hình
        if not ret:  # Nếu không đọc được (cuối video hoặc lỗi)
            print("Video đã chạy hết hoặc không thể đọc thêm khung hình.")
            break

        # Hiển thị khung hình
        cv2.imshow('Khung Hinh', frame)

        # Lưu khung hình vào file
        cv2.imwrite(os.path.join(vdirf, f"Frame{c}.jpg"), frame)
        c += 1

        # Kiểm tra nếu người dùng nhấn phím 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Người dùng dừng video.")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()
    print(f"Tổng số frames đã xử lý: {c}")
    lblfileV.configure(text = "Tổng số famres: %d" %c)
def extract_frames(): 
    global vfile
    if not vfile:
        messagebox.showinfo("Thông báo", f"Đã chọn video mặc định: {os.path.basename(vfile)}")
    if not os.path.exists(os.path.join(vdir, vfile)):
        messagebox.showerror("Lỗi", f"Không tìm thấy video: {vfile}")
        return
    """Extract frames from the selected video.""" 
    # Mở video từ đường dẫn 
    capture = cv2.VideoCapture(os.path.join(vdir, vfile))
    # Đếm số lượng frame 
    frame_count = 0 
    # frame_dir = "frames": Định nghĩa thư mục để lưu các frame. 
    Frame_dir = vdirfacer
    # Tạo thư mục frames nếu chưa tồn tại 
    if not os.path.exists(Frame_dir): 
        os.makedirs(Frame_dir) 
     
    # Extract frames and save as images 
    # Đọc từng frame từ video (cap.read()). 
    while capture.isOpened(): 
        ret, frame = capture.read() 
        if not ret: 
            break 
        frame_path = os.path.join(Frame_dir, f"frame_{frame_count}.jpg") 
        # Lưu frame dưới dạng file ảnh (cv2.imwrite). 
        cv2.imwrite(frame_path, frame) 
        frame_count += 1 
        # Dừng khi đã trích xuất đủ 10 frame hoặc hết video. 
        if frame_count >= 10:  # Limit to 10 frames for simplicity 
            break 
     
    capture.release() 
    # Gọi hàm hiển thị các frame để người dùng chọn. 
    load_frame_selection(Frame_dir) 
 
def load_frame_selection(Frame_dir): 
    """Load extracted frames for user to select.""" 
    # Clear the frame options 
    # Lấy danh sách các widget con trong khung frame_select_frame. 
    for imageidx in frame_select_frame.winfo_children(): 
        imageidx.destroy() 
     
    # Display frames as clickable buttons 
    # os.listdir(frame_dir): Lấy danh sách file trong thư mục frames 
    # sorted(): Sắp xếp danh sách file theo thứ tự chữ cái để đảm bảo thứ tự hiển thị ổn định. 
    # enumerate(): Trả về cả chỉ số (idx) và tên file (frame_file) trong danh sách. 
    # idx: Chỉ số của file trong danh sách, có thể được dùng nếu cần (trong trường hợp này không sử dụng). 
    for idx, frame_file in enumerate(sorted(os.listdir(Frame_dir))): 
        # os.path.join: Tạo đường dẫn đầy đủ tới file bằng cách kết hợp thư mục gốc 
        Frame_path = os.path.join(Frame_dir, frame_file) 
        #  Mở ảnh từ đường dẫn frame_path 
        img = Image.open(Frame_path) 
        img.thumbnail((100, 100))  # Resize for thumbnail display 
        # Chuyển đổi ảnh img (Pillow) thành đối tượng PhotoImage 
        # định dạng cần thiết để hiển thị ảnh trong giao diện Tkinter. 
        photo = ImageTk.PhotoImage(img) 
        # lambda: Tạo một hàm ẩn danh để truyền đối số động. 
        # Khi nút được nhấn, gọi hàm process_frame với đối số là đường dẫn của frame (Frame_path). 
        btnXL = tk.Button(frame_select_frame, image=photo, command=lambda p=Frame_path: process_frame(p)) 
        # Lưu trữ đối tượng photo (ảnh) trong nút nhấn btnXL 
        btnXL.image = photo 
        # pack: Đặt nút nhấn vào khung giao diện frame_select_frame. 
        # side="left": Đặt các nút xếp hàng ngang từ trái sang phải. 
        # padx=5, pady=5: Thêm khoảng cách 5 pixel ở hai bên và phía trên/dưới nút. 
        btnXL.pack(side="left", padx=5, pady=5) 
 
def process_frame(frame_path): 
    """Process the selected frame and count faces.""" 
    # cv2.imread(frame_path): Đọc ảnh từ đường dẫn. 
    frame = cv2.imread(frame_path) 
    # Chuyển ảnh sang grayscale để .............. 
    graypic = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # cv2.CascadeClassifier: Tải mô hình phát hiện khuôn mặt Haar Cascade 
    # Mô hình Haar Cascade: 
    # Là một thuật toán dựa trên phương pháp Cascade of Classifiers,  
    # được sử dụng để phát hiện các vật thể (như khuôn mặt, mắt,  
    # biển số xe, v.v.) trong ảnh hoặc video 
    # Sử dụng các đặc trưng Haar (Haar-like features) để phát hiện các vật thể. 
    # Đặc trưng Haar là các vùng hình chữ nhật trong ảnh, được sử dụng để phân tích mức độ sáng tối giữa các vùng cụ thể. Mỗi đặc trưng Haar được tính toán dựa trên hiệu số tổng giá trị pixel giữa các vùng sáng và vùng tối. 
    # Integral Image: tính tổng các giá trị pixel trong một vùng hình chữ nhật bất kỳ chỉ với 4 phép tính 
    # AdaBoost: chọn ra các đặc trưng có khả năng phân biệt cao nhất giữa khuôn mặt và nền. 
    # Cascade of Classifiers: một chuỗi các bộ phân loại (classifier) hoạt động tuần tự. 
    # + Các classifier đầu tiên: Phát hiện nhanh các vùng không phải khuôn mặt và loại bỏ chúng ngay lập tức 
    # + Các classifier sau: Tiếp tục xử lý sâu hơn trên các vùng nghi ngờ có thể là khuôn mặt. 3 | Page 
 
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
    # detectMultiScale: Phát hiện khuôn mặt trong khung hình 
    # scaleFactor=1.1: Tăng/giảm kích thước cửa sổ tìm kiếm 
    # minNeighbors=5: Số lượng "hàng xóm" cần thiết để xác định khuôn mặt hợp lệ 
    #  tham số minNeighbors là một yếu tố rất quan trọng để xác định mức độ tin cậy khi phát hiện một khuôn mặt 
    # Cách thức hoạt động: Haar Cascade quét qua từng vùng của ảnh ở các kích thước và vị trí khác nhau, nó kiểm tra xem vùng đó có thể là một khuôn mặt hay không. Một khuôn mặt được xác nhận nếu vùng đó nhận được đủ số phiếu "hợp lệ" từ các lần kiểm tra trên vùng lân cận (các "hàng xóm"). 
    # "Hàng xóm" (neighbors): Là các ô (bounding boxes) xung quanh một vùng cụ thể trong ảnh, được kiểm tra có đặc điểm giống khuôn mặt. 
    # Mỗi ô sẽ được xem xét dựa trên các đặc trưng Haar và các bộ phân loại tầng (cascade classifier). 
    # Với minNeighbors=5: 
    # Cần ít nhất 5 ô lân cận đồng ý rằng khu vực này là khuôn mặt. 
    # Kết quả: Chỉ phát hiện khuôn mặt ở các vùng có nhiều bằng chứng (đáng tin cậy hơn). 
    # minSize=(30, 30): Kích thước nhỏ nhất của khuôn mặt cần phát hiện. 
    facesPeople = face_cascade.detectMultiScale(graypic, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
     
    for (x, y, w, h) in facesPeople: 
        # cv2.rectangle: Vẽ hình chữ nhật xung quanh mỗi khuôn mặt được phát hiện. 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
    # Hiển thị số lượng khuôn mặt được phát hiện ngay trên ảnh
    text = f"Faces Detected: {len(facesPeople)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)  # Màu đỏ
    thickness = 2
    text_position = (10, 30)  # Vị trí đặt text trên ảnh (x, y)
    cv2.putText(frame, text, text_position, font, font_scale, font_color, thickness)
    
    # Resize the frame to fit the screen (e.g., 800x600) 
    # Set kích thước ảnh khi nhấp vào với width=800 và height=600 
    screen_width, screen_height = 800, 600 
    frame_resized = cv2.resize(frame, (screen_width, screen_height), 
interpolation=cv2.INTER_AREA) 
     
    # Display the processed frame with detected faces 
    cv2.imshow("Selected Frame - Faces Detected", frame_resized) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
 
    #result_label.config(text=f"Faces Detected in Selected Frame: {len(facesPeople)}") 
def OpenTextFile(): 
    global filepath  # Biến toàn cục
    # Hộp thoại mở thư mục
    filepath = filedialog.askopenfilename(title="Bai mau 1_ Open Text File", 
                                          filetypes=(("Text File (.txt)", "*.txt"), 
                                                     ("CSV File (.csv)", "*.csv")))
    # Mở file có dấu TV encoding="utf-8"
    with open(filepath, "r", encoding="utf-8-sig",  errors="replace") as f1:
        data = f1.read()
        ClearText()
        # Nạp nội dung file vào label text
        lblFileText.configure(state=tk.NORMAL)  # Cho phép sửa text
        lblFileText.insert(tk.END, data)
        lblFileText.configure(state=tk.DISABLED)  # Không cho chỉnh sửa nữa = readonly
def XuLy():
    # Mở file text có dấu TV = encoding="utf-8"
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f1:
        s = ""
        lines_list = []
        # Đọc từng line
        for line in f1:
            # Cắt toàn bộ ký tự khoảng trắng của mỗi dòng
            line = line.strip()
            # Kiểm tra dòng tiếp theo có khác rỗng không
            if line != "":
                kytu = line[0]  # Lấy ký tự đầu tiên
                if "0" <= kytu <= "9": 
                    vitri = line.find(":", 0, -1)
                    if vitri == -1:
                        line = line[0:]  # Lấy vị trí từ 0 đến ký tự \n
                    lines_list.append(line)  # Thêm phần tử vào mảng

        # Sắp xếp phần tử trong mảng list
        lines_list.sort()
        # Đọc phần tử trong mảng và xóa trùng lặp
        s = ""
        key = lines_list[0]
        s += key + "\n"
        dem = 1 
        for i in lines_list[1:]:
            if i != key:
                key = i
                s += i + "\n"
                dem += 1
        
        # Xóa nội dung cũ và cập nhật text
        ClearText()
        lblXuLy.configure(state=tk.NORMAL)  # Cho phép chỉnh sửa text
        lblXuLy.insert(tk.END, s)
        lblXuLy.configure(state=tk.DISABLED)  # Không cho chỉnh sửa, chỉ cho đọc

        # Cập nhật số lượng sinh viên
        lblCount.configure(text="Số lượng SV: %d" % dem)
        file_path = os.path.join(tdir, tfile)
        
        # Sau khi xử lý, lưu file kết quả
        with open(file_path, "w+", encoding="utf-8") as f_new:
            f_new.write(s + "\n" + "Số lượng SV: %d" % dem)
def ClearText(): 
    #Dùng delete cho Text
    lblFileText.delete("1.0", tk.END)
    scroll_y.set(0.0,1.0)
def run_sudoku_solver():
    try:
        run_real_time_sudoku()
    except Exception as e:
        print(f"Error: {e}")
def run_video_sudoku_solver():
    try:
        run_real_time_sudoku()
    except Exception as e:
        print(f"Error: {e}")   
########################################
#B5: LẬP FORM GIAO DIỆN GUI 
########################################
 #Khởi tạo đối tượng 
wn = tk.Tk()
wn.title("08 NGUYỄN XUÂN PHÚC, LỚP_PTITHCM, ĐỒ ÁN HỌC PHẦN: LẬP TRÌNH MULTIMEDIA, 2024")
#Lập Winform 
wn.geometry("1200x900")
wn.resizable(tk.TRUE, tk.TRUE)
# Gán phím tắt Escape cho sự kiện Thoat
wn.bind('<Escape>', Thoat) # Khi nhấn phím Escape, gọi hàm Thoat
t1 = "ĐỒ ÁN HP: LẬP TRÌNH ỨNG DỤNG ĐA PHƯƠNG TIỆN: MULTIMEDIA"
t2 = "SV THỰC HIỆN: 08 NGUYỄN XUÂN PHÚC, KHÓA 2020, HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG TP.HCM"
t3 = "LẬP TRÌNH ỨNG DỤNG MULTIMEDIA TRONG LĨNH VỰC GIÁO DỤC"

#Thiết lập thông tin cho Label tên chức năng trên form (lưu ý : thiêt lập height không dùng lớp ttk)
#lblText1 = tk.Label(wn, text =  t1, background = "yellow", fg = "blue", relief = tk.SUNKEN ,  font = "Times 16", borderwidth = 3, width = 60, height = 3)
lblText1 = tk.Label(wn, text =  t1, fg = "blue", font=("Arial", 18), width = 100, height = 1, anchor='center')
# đặt Label vừa thiết lập thông tin ở trên vào Form wn
lblText1.place(x = 10, y = 10)
#Thiết lập thông tin cho Label tên chức năng trên form (lưu ý : thiêt lập height không dùng lớp ttk)
#lblText2 = tk.Label(wn, text =  t2, background = "yellow", fg = "blue", relief = tk.SUNKEN ,  font = "Times 16", borderwidth = 3, width = 60, height = 3)
lblText2 = tk.Label(wn, text =  t2, fg = "blue", font=("Arial", 14), width = 100, height = 1, anchor='center')
# đặt Label vừa thiết lập thông tin ở trên vào Form wn
lblText2.place(x = 10, y = 50)
#Thiết lập thông tin cho Label tên chức năng trên form (lưu ý : thiêt lập height không dùng lớp ttk)
#lblText3 = tk.Label(wn, text =  t3, background = "yellow", fg = "blue", relief = tk.SUNKEN ,  font = "Times 16", borderwidth = 3, width = 60, height = 3)
lblText3 = tk.Label(wn, text =  t3, fg = "blue", font=("Arial", 14), width = 100, height = 1, anchor='center')
# đặt Label vừa thiết lập thông tin ở trên vào Form wn
lblText3.place(x = 10, y = 90)

#Nhập thời gian chuẩn bị và nhận lệnh = giọng nói
lbTextR = tk.Label(wn, text= "Nhập thời gian chuẩn bị", fg="white", font="Times 16",width= 30 , height=1 , bg= "red")
lbTextR.place(x=10, y =140)

txtSourceR = tk.Entry(wn, width = 30,) # Entry = cho nhập vào 
txtSourceR.place(x = 400, y = 145) 
txtSourceR.insert(0, "3")
 #Lấy giá trị NSD đã nhập vào textbox  

# Lấy giá trị NSD đã nhập vào textbox; cắt spaces dư thừa  
a = txtSourceR.get().strip() 
# Nút chọn audio
btnAudio = tk.Button(wn, text="Chọn Audio", width=10, command=chooseAudio)
btnAudio.place(x=600, y=150)
lbTextS = tk.Label(wn, text= "Nhập thời gian phát lệnh bằng giọng nói", fg="white", font="Times 16",width= 30 , height=1 , bg= "red")
lbTextS.place(x=10, y =170)

txtSourceS = tk.Entry(wn, width = 30, ) # Entry = cho nhập vào 
txtSourceS.place(x = 400, y = 175) 
txtSourceS.insert(0, "5") 
# Nút chọn ảnh
btnChooseImage = tk.Button(wn, text="Chọn Ảnh", width=10, command=choose_image)
btnChooseImage.place(x=600, y=225)
# Nút chuyển ảnh thành ảnh xám
btnGray = tk.Button(wn, text="Ảnh Xám", width=10, command=pGray)
btnGray.place(x=700, y=225)
# Nút đọc ảnh màu
btnColor = tk.Button(wn, text="Đọc Ảnh", width=10, command=pColor)
btnColor.place(x=800, y=225)
# Nút lấy kích thước ảnh
btnColorSize = tk.Button(wn, text="Kích Thước Ảnh", width=14, command= pColorSize)
btnColorSize.place(x=900, y=225)
# Nút cắt ảnh 
btnCutPart = tk.Button(wn, text="Cắt Ảnh", width=10, command= pColorPart)
btnCutPart.place(x=1030, y=225)
# Nút chức năng quay hình
btnRotate = tk.Button(wn, text="Quay Ảnh", width=10, command= pRotate)  
btnRotate.place(x=600, y=275)
# Nút chức năng resize
btnResize = tk.Button(wn, text="Resize Ảnh", width=10, command= pResize)  
btnResize.place(x=700, y=275)
# Nút chức năng kích thước màu
btnColorRGB = tk.Button(wn, text="Màu Ảnh", width=10, command= getRGBPixel)
btnColorRGB.place(x=800, y=275)

# Lấy giá trị NSD đã nhập vào textbox; cắt spaces dư thừa  
aS = txtSourceS.get().strip() 
btnChooseVideo = tk.Button(wn, text="Chọn Video", width=10, command=choose_video)
btnChooseVideo.place(x=600, y=325)
# Nút để cắt video thành các frame
btnCutVideo = tk.Button(wn, text="Cắt Video", width=10, command= cut_video_frames)
btnCutVideo.place(x=700, y=325)
# Button CẮT FRAME TỪ VIDEO
btnFrVideo = tk.Button(wn, text = "Cắt Frame", width = 10,command = VideoFrame)
btnFrVideo.place(x = 800, y = 325)
# Nút nhận diện khuông mặt
btnFaceVideo = tk.Button(wn, text = "Face Reg", width= 10, command= extract_frames)
btnFaceVideo.place(x = 900, y = 325)
frame_select_frame = tk.Frame(wn) 
frame_select_frame.place(x =20, y=375) 
# Thiết lập nút Open file
btnOpenTextFile = tk.Button(wn, text="Open text File", width= 10, command=OpenTextFile)
btnOpenTextFile.place(x=480, y=500)
# Thiết lập nút Xử lý
btnXuLy = tk.Button(wn, text="Xử lý", width= 10, command=XuLy)
btnXuLy.place(x=480, y=550)
#Real time sudoku
btnRunSudoku = tk.Button(wn, text="Sudoku realtime", width=15, command=run_sudoku_solver)
btnRunSudoku.place(x=600, y=450)
#Video sudoku
btnVideoSudoku = tk.Button(wn, text="Sudoku video", width=15, command=run_video_sudoku_solver)
lblfileA = tk.Label(wn,text="", relief = tk.SUNKEN)
lblfileA.place(x=1000, y=150)
lblfileP = tk.Label(wn,text="", relief = tk.SUNKEN)
lblfileP.place(x=20, y=225)
lblfileV = tk.Label(wn,text="", relief = tk.SUNKEN)
lblfileV.place(x=20, y=325)

 # Button THOÁT 
btnThoat = tk.Button(wn, text = "Thoát", width = 10, command = Thoat)
btnThoat.place(x = 1100, y = 850) # căn cứ vào kích thước form [wn.geometry("800x600")] => canh vị trí Button "thoát"
# Button LỆNH 
btnLenh = tk.Button(wn, text = "Lệnh = nói", width = 10,command = Lenh)
btnLenh.place(x = 700, y = 150)
# Button ĐỌC
btnDoc = tk.Button(wn, text = "Đọc", width = 10,command = Doc)
btnDoc.place(x = 800, y = 150)
# Button DỊCH
btnDoc = tk.Button(wn, text = "Dịch", width = 10,command = translate_audio_with_playback)
btnDoc.place(x = 900, y = 150)
##############################################
#Thiết lập 1 frame chứa thông tin 
##############################################
# Thiết lập frame chứa thông tin file text
frame = tk.Frame(wn, width=380, height=300, relief=tk.SUNKEN, borderwidth=3)
frame.place(x=20, y=500)
# Thiết kế label để đọc file text có chứa scroll
lblFileText = tk.Text(frame, width=50, state=tk.DISABLED)
scroll_y = tk.Scrollbar(frame, command=lblFileText.yview, orient=tk.VERTICAL)
lblFileText.configure(yscrollcommand=scroll_y.set)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
lblFileText.pack(side=tk.LEFT, fill=tk.BOTH)

# Thiết lập frame2 chứa thông tin xử lý file text
frame2 = tk.Frame(wn, width=380, height=300, relief=tk.SUNKEN, borderwidth=3)
frame2.place(x=600, y=500)

# Thiết kế label để đọc file text có chứa scroll
lblXuLy = tk.Text(frame2, width=50, state=tk.DISABLED)
scroll_y = tk.Scrollbar(frame2, command=lblXuLy.yview, orient=tk.VERTICAL)
lblXuLy.configure(yscrollcommand=scroll_y.set)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
lblXuLy.pack(side=tk.LEFT, fill=tk.BOTH)
# Thiết lập label đếm số lượng sinh viên
lblCount = tk.Label(wn, text="Số lượng SV: 0", relief=tk.SUNKEN, width=12)
lblCount.place(x=476, y=600)

wn.mainloop()
