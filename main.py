from kivy.app import App
from kivy.core.text import LabelBase
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
import arabic_reshaper
from bidi.algorithm import get_display
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.core.audio import SoundLoader
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import os, glob, random
from scipy.signal import resample
import soundfile as sf
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from jnius import autoclass
import os

# دریافت کلاس‌های مورد نیاز از اندروید
Environment = autoclass('android.os.Environment')

data_out = {'شماره آزمون':[],
            'منبع صدا':[],
            'کاراکتر':[],
            'جهت پخش صدا':[],
            'سرعت پخش':[],
            'مدت زمان (ثانیه)':[],
            'فاصله':[]}

def output_csv(name):  
    df = DataFrame(data_out) 
    os.makedirs('output_CSV', exist_ok=True)  
    dir_file = os.path.join(name,'data file.csv')   

    # ذخیره‌سازی در فایل CSV  
    df.to_csv(dir_file, index=False, encoding='utf-8-sig')   

def process_sounds(output_folder):
    def adjust_volume(signal, current_time, total_time, ref_distance=1.0, ref_level=0.0):
        # محاسبه فاصله (از ۱۰ متر به ۰ متر به صورت خطی)
        distance = 10 * (1 - current_time / total_time)
        
        # محاسبه تضعیف (attenuation) بر اساس فاصله
        attenuation = 20 * np.log10(distance / ref_distance)
        
        # محاسبه ضریب مقیاس (scale factor)
        scale_factor = 10 ** ((ref_level - attenuation) / 20)
        
        # اعمال تغییرات حجم صدا
        return signal * scale_factor

    direct_mapping = {
        '0 deg': np.ones(10, dtype=np.int64) * 18,
        '45 deg': np.arange(9, -1, -1),
        '90 deg': np.ones(10, dtype=np.int64) * 0,
        '135 deg': np.arange(9, -1, -1),
        '180 deg': np.ones(10, dtype=np.int64) * 18,
        '225 deg': np.arange(27, 37, 1),
        '270 deg': np.ones(10, dtype=np.int64) * 36,
        '315 deg': np.arange(27, 37, 1),}

    speed_values = [0.5, 1.0, 1.5, 2.0]

    kemar_path = os.path.join("kemar","elev-10")
    source_path = 'source'
    Kemar = glob.glob(os.path.join(kemar_path, '*.wav'))
    Source = glob.glob(os.path.join(source_path, '*.wav'))
    Kemar.sort()

    # پردازش برای هر سرعت و هر فایل منبع
    for sp in speed_values:
        for sor in Source:
            # خواندن فایل منبع
            [sig, fs_s] = sf.read(sor)
            num_parts = 10
            length = len(sig)
            part_size = length // num_parts
            total_time = length / fs_s
            
            # تقسیم سیگنال به ۱۰ بخش
            segments = [sig[i * part_size:(i + 1) * part_size] for i in range(num_parts)]

            # پردازش برای هر جهت
            for deg, values in direct_mapping.items():
                mix = np.array([])
                (a, b) = (1, 0) if deg in ['135 deg', '180 deg', '225 deg'] else (0, 1)

                counter = 0
                for val in values:
                    # خواندن فایل HRTF
                    [HRTF, fs_H] = sf.read(Kemar[val])
                    current_time = (counter / num_parts) * total_time

                    # کانولوشن سیگنال با HRTF
                    s_L = np.convolve(segments[counter], HRTF[:, a], mode='full')
                    s_R = np.convolve(segments[counter], HRTF[:, b], mode='full')

                    # تنظیم حجم صدا بر اساس فاصله
                    s_R = adjust_volume(s_R, current_time, total_time)
                    s_L = adjust_volume(s_L, current_time, total_time)

                    # ترکیب سیگنال‌های چپ و راست
                    mix_n = np.vstack([s_L, s_R]).T

                    # اضافه کردن به میکس نهایی
                    mix = mix_n if mix.size == 0 else np.concatenate((mix, mix_n), axis=0)
                    counter += 1

                # تغییر سرعت سیگنال
                num_samples = int(len(mix) / sp)
                mix_sig = resample(mix, num_samples)
                mix_sig = mix_sig / np.max(np.abs(mix_sig))  # نرمال‌سازی

                # ذخیره فایل خروجی
                output_name = f"{os.path.splitext(os.path.basename(sor))[0]}_{deg}_{sp}x.wav"
                output_path = os.path.join(output_folder, output_name)
                sf.write(output_path, mix_sig, fs_s, format='WAV', subtype='PCM_16')

def plot_sound_path_polar(stop_time, direction, speed, path , num):
 
    total_time = 10/speed
    stop_distance = 10 * (1 - stop_time / total_time)
    data_out["فاصله"].append(stop_distance)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    
    direction_rad = np.deg2rad(int(direction.split()[0]))   # جهت صدا (تبدیل به رادیان)
    
    circle_radius = 10  # محیط دایره
    theta = np.linspace(0, 2 * np.pi, 100)  # زاویه‌ها از ۰ تا ۳۶۰ درجه
    ax.plot(theta, [circle_radius] * len(theta), linestyle='--', linewidth=0.5,color='black')
    
    ax.plot([direction_rad, direction_rad], [circle_radius, stop_distance],  linestyle='--', linewidth=2,color='red')
    
    ax.scatter(direction_rad, stop_distance, color='red', s=110)
    
    ax.set_theta_zero_location('E')  # جهت صفر درجه به سمت بالا
    ax.set_theta_direction(1)  # جهت چرخش ساعت‌گرد
    ax.set_rlabel_position(0)  # تنظیم برچسب فاصله
    ax.set_rmax(10)  # حداکثر فاصله (محیط دایره)
    ax.set_rticks(np.arange(0, 11, 1))  # تقسیم‌بندی فاصله
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.gcf().set_facecolor('#2b4352')
    ax.set_facecolor('lightgray')
  
    image_path = os.path.join(path, "polt task")
    os.makedirs(image_path,exist_ok=True)
    image_path_task = os.path.join(image_path, "task "+ num+".png")
    plt.savefig(image_path_task)
    plt.close()  # بستن نمودار برای آزادسازی منابع

def plot_grouped_density(data, path , group_col, value_col, selected_groups=None):
    """
    رسم نمودار چگالی برای هر گروه به‌صورت جداگانه، با امکان فیلتر گروه‌ها
    
    پارامترها:
    - data: دیتافریم ورودی
    - group_col: نام ستونی که گروه‌ها را مشخص می‌کند (مثلاً "منبع صدا" یا "کاراکتر" یا "شماره آزمون")
    - value_col: نام ستونی که مقدار عددی برای رسم چگالی دارد (مثلاً "فاصله")
    - selected_groups: لیستی از گروه‌های خاص برای نمایش (اگر None باشد، همه گروه‌ها نمایش داده می‌شوند)
    """
    # بررسی وجود ستون‌ها
    if group_col not in data.columns or value_col not in data.columns:
        raise ValueError("ستون‌های داده معتبر نیستند!")

    # حذف مقدارهای NaN
    data = data.dropna(subset=[group_col, value_col])

    # اگر گروه‌های خاصی مشخص شده باشند، فقط همان‌ها را نمایش بده
    if selected_groups:
        data = data[data[group_col].isin(selected_groups)]

    # استخراج گروه‌های یکتا
    unique_groups = data[group_col].unique()
    
    if len(unique_groups) == 0:
        print("هیچ گروهی برای نمایش وجود ندارد!")
        return

    # اگر گروه‌بندی بر اساس "شماره آزمون" باشد، تنها یک نمودار رسم شود
    if group_col == "شماره آزمون":
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=data, x=value_col, fill=True, color='blue')
        plt.title("Density Map")
        plt.ylabel("Density")
        plt.xlabel(value_col)
        image_path_task = os.path.join(path, "plot_grouped_density _ "+ group_col+".png")
        plt.savefig(image_path_task)
        plt.close()
        return

    # تنظیم تعداد ساب‌پلات‌ها بر اساس تعداد گروه‌ها
    fig, axes = plt.subplots(len(unique_groups), 1, figsize=(8, 2 * len(unique_groups)), sharex=True)

    if len(unique_groups) == 1:
        axes = [axes]

    palette = sns.color_palette("husl", len(unique_groups))

    for i, group in enumerate(unique_groups):
        ax = axes[i]
        subset = data[data[group_col] == group]
        sns.kdeplot(data=subset, x=value_col, fill=True, color=palette[i], ax=ax)
        ax.set_title(f"Density Map for {group}")
        ax.set_ylabel("Density")
    
    plt.xlabel(value_col)
    plt.tight_layout()

    image_path_task = os.path.join(path, "plot_grouped_density _ "+ group_col+".png")
    plt.savefig(image_path_task)
    plt.close()

def plot_polar_sound_sources(data,path, group_col="منبع صدا", selected_groups=None):
    """
    رسم نمودار قطبی برای منابع صدا یا کاراکترها
    
    پارامترها:
    - data: دیتافریم ورودی
    - group_col: نام ستونی که گروه‌ها را مشخص می‌کند (مثلاً "منبع صدا" یا "کاراکتر" یا "شماره آزمون")
    - selected_groups: لیستی از گروه‌های خاص برای نمایش (اگر None باشد، همه گروه‌ها نمایش داده می‌شوند)
    """
    # بررسی وجود ستون‌ها
    if "جهت پخش صدا" not in data.columns or "فاصله" not in data.columns or group_col not in data.columns:
        raise ValueError("ستون‌های لازم در داده‌ها موجود نیستند!")

    # حذف مقدارهای NaN در صورت وجود
    df = data.dropna(subset=["جهت پخش صدا", "فاصله", group_col])

    # تبدیل "جهت پخش صدا" به عددی و سپس به رادیان
    df["جهت پخش صدا"] = pd.to_numeric(df["جهت پخش صدا"], errors="coerce")
    df = df.dropna(subset=["جهت پخش صدا"])  # حذف مقدارهای نامعتبر
    angles = np.radians(df["جهت پخش صدا"])
    radii = df["فاصله"]
    sources = df[group_col]  # انتخاب ستون گروه‌بندی‌شده

    # اگر گروه‌های خاصی انتخاب شده باشند، فقط آن‌ها را نمایش دهیم
    if selected_groups:
        df = df[df[group_col].isin(selected_groups)]
        angles = np.radians(df["جهت پخش صدا"])
        radii = df["فاصله"]
        sources = df[group_col]

    # بررسی اینکه آیا گروه‌بندی بر اساس "شماره آزمون" است؟
    single_color = (group_col == "شماره آزمون")

    # تعریف رنگ‌های مشخص برای هر گروه (در صورتی که مبنا شماره آزمون نباشد)
    if not single_color:
        unique_groups = df[group_col].unique()
        color_palette = plt.cm.get_cmap("tab10", len(unique_groups))
        group_colors = {group: color_palette(i) for i, group in enumerate(unique_groups)}
        colors = [group_colors.get(src, 'black') for src in sources]
    else:
        colors = ['blue'] * len(df)  # همه نقاط یک رنگ خواهند داشت

    # رسم نمودار قطبی
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(angles, radii, c=colors, alpha=0.75)
    ax.set_ylim(0, 10)  # تنظیم محدوده شعاع
    ax.set_title(f"نمودار قطبی برای {group_col}")

    # ایجاد لیبل رنگ‌ها (Legend) در صورتی که مبنا شماره آزمون نباشد
    if not single_color:
        legend_patches = [mpatches.Patch(color=color, label=label) for label, color in group_colors.items()]
        ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    image_path_task = os.path.join(path, "plot_polar_sound_sources _"+ group_col +".png")
    plt.savefig(image_path_task)
    plt.close()


LabelBase.register(name='font', fn_regular='font/STITRBD.ttf')

class BaseScreen(Screen):
    def reshape_text(self, text):
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text
    
    def show_popup(self, message):
        # نمایش پاپ‌آپ با پیام خطا
        popup = Popup(
            title="Error",
            content=Label(text=self.reshape_text(message)),
            size_hint=(0.5, 0.5)
        )
        popup.open()

class MainWin(BaseScreen):
    def on_enter(self):
        folder_out = "Patient Output"
        os.makedirs(folder_out, exist_ok=True)

    def btn(self):
        if self.ids.name.text =="":
            self.show_popup("اسم نمیتواند خالی باشد")
        else:
            default_output_folder = os.path.join(self.manager.app_storage_path, 'output_sounds')
            if not os.path.exists(default_output_folder):
                os.makedirs(default_output_folder)

            # بررسی تعداد فایل‌های خروجی
            output_sound = glob.glob(os.path.join(default_output_folder, '*.wav'))
            if len(output_sound) != 192:  # اگر فایل‌ها وجود ندارند، پردازش انجام شود
                process_sounds(default_output_folder)
                output_sound = glob.glob(os.path.join(default_output_folder, '*.wav'))

            Patient_name = self.ids.name.text
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H-%M")
            path_Patient = Patient_name + " "+ formatted_time

            downloads_path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath()
            folder_patient=os.path.join(downloads_path,"Patient Output")
            os.makedirs(folder_patient,exist_ok=True)
            self.manager.patient_name_path = os.path.join(folder_patient, path_Patient)
            os.makedirs(self.manager.patient_name_path)
            self.manager.transition.direction = "left"  # تعیین جهت انیمیشن
            self.manager.current = "settings"  # تغییر صفحه

class SettingsPage(BaseScreen):
    def combine_lists(self, list1, list2, list3, n):
        random.shuffle(list3)
        result = []
        last_a = None  # آخرین مقدار انتخاب‌شده از list1
        
        for _ in range(n):
            a_choices = [a for a in list1 if a != last_a]  # جلوگیری از تکرار پشت سر هم در list1
            if not a_choices:
                a_choices = list1  # در صورت نیاز، لیست را مجدداً مجاز کن
            a = random.choice(a_choices)
            last_a = a  # به‌روزرسانی مقدار آخرین انتخاب

            b = random.choice(list2)
            c = random.choice(list3)  # بدون محدودیت برای مقدار c

            result.append(f"{a}_{b}_{c}.wav")
            
        return result

    selected_options_Suorce = []
    def SuorceBox(self, checkbox, value, option_id):
        options_map = {
            1: "Child",
            2: "Man formal",
            3: "Man informal",
            4: "Neutral",
            5: "Woman formal",
            6: "Woman informal"
        }

        option_text = options_map.get(option_id, "نامشخص")

        if value:
            self.selected_options_Suorce.append(option_text)  # اضافه کردن گزینه به لیست
        else:
            self.selected_options_Suorce.remove(option_text)


    selected_options_Ori = []
    def OriBox(self, checkbox, value, option_id):
        options_map = {
            1: "0 deg",
            2: "45 deg",
            3: "90 deg",
            4: "135 deg",
            5: "180 deg",
            6: "225 deg",
            7: "270 deg",
            8: "315 deg"
        }

        option_text = options_map.get(option_id, "نامشخص")

        if value:
            # بررسی اینکه تعداد انتخاب‌ها از 3 بیشتر نشود
            if len(self.selected_options_Ori) >= 4:
                checkbox.active = False  # غیرفعال کردن انتخاب جدید
                self.show_popup("شما نمی‌توانید بیش از چهار گزینه انتخاب کنید.")
            else:
                self.selected_options_Ori.append(option_text)  # اضافه کردن گزینه به لیست
        else:
            # حذف گزینه از لیست اگر غیرفعال شد
            if option_text in self.selected_options_Ori:
                self.selected_options_Ori.remove(option_text)

    selected_options_Speed =[]
    def SpeedBox(self, checkbox, value, option_id):
        options_map = {
            1: "0.5x",
            2: "1.0x",
            3: "1.5x",
            4: "2.0x",
        }

        option_text = options_map.get(option_id, "نامشخص")
        if value:
            self.selected_options_Speed.append(option_text)  # اضافه کردن گزینه به لیست
        else:
            self.selected_options_Speed.remove(option_text)
    def combine_list_string(self,lst, string):

        return [f"{item}*{string}" for item in lst]

    def btn(self):
        sel_Source=[]
        if any([
            self.ids.task_number.text in ["0", ""], not self.selected_options_Ori, 
            not self.selected_options_Speed, not self.selected_options_Suorce]):
            self.show_popup("هیچ کدام از مقادیر نمی تواند خالی باشد ")
        elif len(self.selected_options_Ori) < 4:
            self.show_popup("باید چهار جهت برای آزمون")
        else:
            for elm in self.selected_options_Suorce:
                # استفاده از یک دیکشنری برای نگهداری متناظر elm با self.ids.sorX
                source_mapping = {
                    "Child": self.ids.sor1,
                    "Man formal": self.ids.sor2,
                    "Man informal": self.ids.sor3,
                    "Neutral": self.ids.sor4,
                    "Woman formal": self.ids.sor5,
                }
                
                # اگر elm در دیکشنری وجود داشت، از آن استفاده کن، در غیر این صورت از sor6
                source_widget = source_mapping.get(elm, self.ids.sor6)
                
                # دریافت متن و تقسیم آن به لیست
                textinpulist = source_widget.text.split("\n")
                
                # ترکیب لیست با elm
                flist = self.combine_list_string(textinpulist, elm)
                
                # اضافه کردن عناصر flist به sel_Source (نه کل لیست flist)
                sel_Source.extend(flist)

            n = int(self.ids.task_number.text)
            self.manager.list_sound = self.combine_lists(
                sel_Source, 
                self.selected_options_Ori,
                self.selected_options_Speed, 
                n
            )
            print(self.manager.list_sound)
            self.manager.current = "start"

class StartPage(BaseScreen):
    def btn(self):
        self.manager.current = "first"

class FirstPage(BaseScreen):
    img = ObjectProperty(None)
    def on_enter(self):
        spl_list = self.manager.list_sound[0].split("*")
        
        esm = spl_list[0]
        self.manager.caracter = esm
        self.manager.list_sound[0] = spl_list[1]
        loc_list = self.manager.list_sound
        print(loc_list)
        deg_sor = loc_list[0].split("_")[1]
        self.img.source = "images/" +deg_sor +".png"
        self.ids.my_label.text = self.reshape_text(esm)
  
    is_unlocked = False  
    attempts = 0  
    max_attempts = 5  # حداکثر تلاش مجاز

    def show_password_popup(self):
        if not self.is_unlocked and self.attempts < self.max_attempts:
            self.popup = Popup(title="Password Required", size_hint=(0.5, 0.4), auto_dismiss=False)
            
            layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
            layout.add_widget(Label(text=self.reshape_text(" رمز عبور را وارد کنید:") , font_name='font'))
            
            self.password_input = TextInput(hint_text="Enter password", password=True, multiline=False)
            layout.add_widget(self.password_input)
            
            button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height="40dp", spacing=10)
            close_btn = Button(text=self.reshape_text("بستن"), font_name='font', on_press=self.close_popup)
            submit_btn = Button(text=self.reshape_text("ادامه"), font_name='font', on_press=self.check_password)
            button_layout.add_widget(close_btn)
            button_layout.add_widget(submit_btn)
            
            layout.add_widget(button_layout)
            self.popup.content = layout
            self.popup.open()

    def check_password(self, instance):
        if self.password_input.text == "sbu123":  # رمز صحیح
            self.popup.dismiss()
            self.on_button_click()
        else:
            self.attempts += 1  # افزایش تعداد تلاش‌های ناموفق
            self.password_input.text = ""
            self.password_input.hint_text = f"Incorrect! {self.max_attempts - self.attempts} attempts left"

            if self.attempts >= self.max_attempts:  # اگر تعداد تلاش‌ها تمام شد
                self.popup.dismiss()
                self.disable_button()

    def close_popup(self, instance):
        self.popup.dismiss()

    def disable_button(self):
        self.ids.main_buttonl.text = self.reshape_text("غیر فعال شد")
        self.ids.main_button.disabled = True

    def on_button_click(self):
        self.manager.current = "last"

    def btn(self):
        self.manager.current = "contorl"
        self.manager.transition.direction = "left"
        data_out["کاراکتر"].append(self.manager.caracter )

class ContorolPage(BaseScreen): 
    elapsed_time = 0.0  # مقدار ثانیه‌شمار
    running = False  # وضعیت تایمر

    def on_enter(self):
        loc_list = self.manager.list_sound
        direct = os.path.join(self.manager.app_storage_path,"output_sounds", loc_list[0] )  
        self.sound = SoundLoader.load(direct)

    def start_timer(self):
        if not self.running:
            self.running = True
            Clock.schedule_interval(self.update_time, 0.01) 
            self.sound.play()
            self.ids.button1.opacity = 0  # Hide Button 1
            self.ids.button1.disabled = True  # Disable Button 1
            self.ids.button2.opacity = 1  # Show Button 2
            self.ids.button2.disabled = False  # Enable Button 2
            self.ids.lab2.text = self.reshape_text('توقف')
            self.ids.lab2.color = (170/255, 193/255, 209/255,1)
            self.ids.lab2.pos_hint = {'center_x':0.5, 'center_y':0.6}

    def stop_timer(self):
        if self.running:
            self.running = False
            Clock.unschedule(self.update_time)
            self.sound.stop()
            self.save_time()

    def update_time(self, dt):
        self.elapsed_time += dt

    def save_time(self):
        if not self.running:
            stop_time = round(self.elapsed_time, 2)
            self.manager.number_task +=1
            self.elapsed_time = 0.0
            loc_list =self.manager.list_sound[0].split("_")
            print(loc_list)
            direction = loc_list[1].split(" ")[0]
            speed = float(loc_list[2][:-5])
            print(direction)
            print(speed)
            data_out["مدت زمان (ثانیه)"].append(stop_time)
            data_out["سرعت پخش"].append(speed)
            data_out["جهت پخش صدا"].append(direction)
            data_out["منبع صدا"].append(loc_list[0])
            data_out["شماره آزمون"].append(self.manager.number_task)
            self.Initial_state()
            number = str(self.manager.number_task)
            plot_sound_path_polar(stop_time, direction, speed,
                                   self.manager.patient_name_path,number)
            if len(self.manager.list_sound)==0:
                self.manager.current = "end"
            else:
                self.manager.current = "first"
    
    def Initial_state(self):
            self.ids.button1.opacity = 1  # Hide Button 1
            self.ids.button1.disabled = False  # Disable Button 1
            self.ids.button2.opacity = 0  # Show Button 2
            self.ids.button2.disabled = True  # Enable Button 2
            self.ids.lab2.text = self.reshape_text('شروع')
            self.ids.lab2.color = (5/255, 29/255, 53/255, 1)
            self.ids.lab2.pos_hint = {'center_x':0.5, 'center_y':0.4}
            del self.manager.list_sound[0]

class EndPage(BaseScreen):
    is_unlocked = False  
    attempts = 0  
    max_attempts = 5  # حداکثر تلاش مجاز

    def show_password_popup(self):
        if not self.is_unlocked and self.attempts < self.max_attempts:
            self.popup = Popup(title="Password Required", size_hint=(0.5, 0.4), auto_dismiss=False)
            
            layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
            layout.add_widget(Label(text=self.reshape_text(" رمز عبور را وارد کنید:") , font_name='font'))
            
            self.password_input = TextInput(hint_text="Enter password", password=True, multiline=False)
            layout.add_widget(self.password_input)
            
            button_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height="40dp", spacing=10)
            close_btn = Button(text=self.reshape_text("بستن"), font_name='font', on_press=self.close_popup)
            submit_btn = Button(text=self.reshape_text("ادامه"), font_name='font', on_press=self.check_password)
            button_layout.add_widget(close_btn)
            button_layout.add_widget(submit_btn)
            layout.add_widget(button_layout)
            self.popup.content = layout
            self.popup.open()

    def check_password(self, instance):
        if self.password_input.text == "sbu123":  # رمز صحیح
            self.popup.dismiss()
            self.on_button_click()
        else:
            self.attempts += 1  # افزایش تعداد تلاش‌های ناموفق
            self.password_input.text = ""
            self.password_input.hint_text = f"Incorrect! {self.max_attempts - self.attempts} attempts left"

            if self.attempts >= self.max_attempts:  # اگر تعداد تلاش‌ها تمام شد
                self.popup.dismiss()
                self.disable_button()

    def close_popup(self, instance):
        self.popup.dismiss()

    def disable_button(self):
        self.ids.main_buttonl.text = self.reshape_text("غیر فعال شد")
        self.ids.main_button.disabled = True

    def on_button_click(self):
        self.manager.current = "last"

class LastPage(BaseScreen):
    def save_file(self):
        output_csv(self.manager.patient_name_path)
        df = pd.DataFrame(data_out)
        plot_grouped_density(df, self.manager.patient_name_path,group_col='منبع صدا', value_col="فاصله")
        plot_grouped_density(df, self.manager.patient_name_path,group_col='کاراکتر', value_col="فاصله")
        plot_grouped_density(df, self.manager.patient_name_path,group_col="شماره آزمون", value_col="فاصله")
        plot_polar_sound_sources(df, self.manager.patient_name_path, group_col='منبع صدا')
        plot_polar_sound_sources(df, self.manager.patient_name_path, group_col='کاراکتر')
        plot_polar_sound_sources(df, self.manager.patient_name_path, group_col="شماره آزمون")

    def save_exit(self):
        self.save_file()
        App.get_running_app().stop()

    def save_continue(self):
        self.save_file()
        self.manager.number_task = 0
        data_out["شماره آزمون"]= []
        data_out["جهت پخش صدا"] = []
        data_out["سرعت پخش"] = []
        data_out["فاصله"] = []
        data_out["مدت زمان (ثانیه)"] = []
        data_out["منبع صدا"]= [] 
        self.manager.transition.direction = "right"
        self.manager.current = "main"

    def exitAPP(self):
        App.get_running_app().stop()

class windowsmanager(ScreenManager):
    list_sound = ObjectProperty([])
    app_storage_path = App.get_running_app().user_data_dir
    patient_name_path = "" 
    number_task = 0
    caracter = ''

kv = Builder.load_file('main.kv')
class AuditoryTrustTest(App):
    def build(self):
        self.icon="images/icon.png"
        Window.size = (1040, 650)
        return kv

if __name__ == "__main__":
    AuditoryTrustTest().run()