import math
import random
import shutil

import pygame
import sys
import os
import csv
from datetime import datetime  # 用于记录时间戳
from pupilio import Pupilio, DefaultConfig
from typing import List, Dict, Tuple, Optional

class SingleImageTrialManager:
    """单张情绪图片试次管理器（含试次图片记录功能）"""

    GREY = (128, 128, 128)
    BLACK = (0, 0, 0)

    DOT_RADIUS = 10
    
    WIDTH = 1920
    HEIGHT = 1080
    WHITE_CIRCLE_CENTER = (WIDTH // 2, HEIGHT // 2)
    GRAY = (150, 150, 150)
    WHITE = (255, 255, 255)

    BUTTON_BLUE = (173, 216, 230)

    CORNER_RADIUS = 30
    INNER_PX = 60
    
    FPS = 30

    def __init__(self, sub_info, model=0):

        self.model = model
        
        # 1. 基础配置（与原逻辑一致）
        self.pi = None  # 眼动设备实例，初始化时设为None
        self.pygame_initialized = False  # 标记Pygame是否已初始化
        # 2. 初始化Pygame
        pygame.init()
        self.pygame_initialized = True

        # 1. 获取当前模块的绝对路径（无论单独运行还是包导入，均生效）
        current_module = os.path.abspath(__file__)
        # 2. 获取包根目录（main_module.py 所在的目录）
        self.package_root = os.path.dirname(current_module)
        # # 3. 拼接目标文件路径（不受当前工作目录影响）
        # target_path = self.package_root / "fonts" / "msyh.ttc"

        self.sub_info = sub_info
        _sub_id = f"{sub_info['name']}_{sub_info['gender']}_{sub_info['age']}"
        self.save_dir = os.path.join(self.package_root, 'data', f"{_sub_id}_{sub_info['now']}")
        os.makedirs(self.save_dir, exist_ok=True)

        # 情绪图片路径
        self.emotion_folder = os.path.join(self.package_root, "images")

        # 4. 加载txt试次参数
        csv_path = os.path.join(self.package_root, "TrialList", "trials_list.csv")
        self.trials_list = self._load_trials_from_csv(csv_path)

        # 加载字体
        self.title_font = self._load_font(80, is_bold=True)
        self.text_font = self._load_font(36)
        self.button_font = self._load_font(32, is_bold=True)
        self.font_btn = self._load_font(30, is_bold=True)

        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("单张情绪图片试次（含记录功能）")
        self.clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        # print("🖱️  试次启动，隐藏鼠标显示")

        # 3. 初始化试次记录文件（CSV格式，便于后续分析）
        self.record_file_path = os.path.join(self.save_dir, "trial_image_records.csv")
        self._init_trial_record()

        # 图片内容
        self.images = os.path.join(self.save_dir, 'images')
        os.makedirs(self.images, exist_ok=True)


        self.trigger_csv = os.path.join(self.package_root, "TrialList", "情绪定向任务_trigger.csv")

        #
        self.receive_statues = "任务未完成"
        # 启动时立即置顶窗口
        self.set_window_topmost(True)

    def set_window_topmost(self, topmost=True):
        """设置Pygame窗口置顶（跨平台支持）"""
        try:
            if sys.platform == "win32":
                # Windows系统：用pywin32控制窗口
                import win32gui
                import win32con
                # 获取Pygame窗口句柄
                hwnd = win32gui.FindWindow(None, pygame.display.get_caption()[0])
                if hwnd:
                    # 设置窗口置顶（SWP_NOMOVE/SWP_NOSIZE 保持窗口位置和大小不变）
                    win32gui.SetWindowPos(
                        hwnd,
                        win32con.HWND_TOPMOST if topmost else win32con.HWND_NOTOPMOST,
                        0, 0, 0, 0,
                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
                    )
            # elif sys.platform == "darwin":
            #     # Mac系统：用pyobjc控制窗口
            #     from AppKit import NSApplication, NSWindow
            #     app = NSApplication.sharedApplication()
            #     for window in app.windows():
            #         if window.title() == pygame.display.get_caption()[0]:
            #             window.setLevel_(NSWindow.Level.floating)  # 置顶
            #             break
            # elif sys.platform.startswith("linux"):
            #     # Linux系统：用wmctrl命令（需系统安装wmctrl：sudo apt install wmctrl）
            #     import os
            #     window_title = pygame.display.get_caption()[0]
            #     cmd = f"wmctrl -F -r '{window_title}' -b add,above" if topmost else f"wmctrl -F -r '{window_title}' -b remove,above"
            #     os.system(cmd)
            print(f"窗口置顶状态：{'开启' if topmost else '关闭'}")
        except Exception as e:
            print(f"设置窗口置顶失败：{str(e)}")

    def _load_font(self, size, is_bold=False):
        """加载字体（优先本地， fallback系统字体）"""
        font_filename = "msyhbd.ttc" if is_bold else "msyh.ttc"
        font_path = os.path.join(self.package_root, "fonts", font_filename)
        try:
            return pygame.font.Font(font_path, size)
        except:
            print(f"字体加载失败，使用默认{('粗体' if is_bold else '常规')}字体")
            return pygame.font.SysFont(["Microsoft YaHei", "SimHei"], size, bold=is_bold)

    # ---------------------- 图形绘制方法 ----------------------
    def _draw_inner_area(self) -> pygame.Rect:
        inner_rect = pygame.Rect(self.INNER_PX, self.INNER_PX, self.WIDTH - 2 * self.INNER_PX,
                                 self.HEIGHT - 2 * self.INNER_PX)
        self._draw_rounded_rect(self.screen, self.WHITE, inner_rect, self.CORNER_RADIUS)
        return inner_rect

    @classmethod
    def _draw_rounded_rect(cls, surface: pygame.Surface, color: Tuple[int, int, int],
                           rect: pygame.Rect, corner_radius: int) -> None:
        if corner_radius < 0:
            corner_radius = 0
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius, rect.top + corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.left + corner_radius, rect.bottom - corner_radius), corner_radius)
        pygame.draw.circle(surface, color, (rect.right - corner_radius, rect.bottom - corner_radius), corner_radius)
        pygame.draw.rect(surface, color,
                         (rect.left, rect.top + corner_radius, rect.width, rect.height - 2 * corner_radius))
        pygame.draw.rect(surface, color,
                         (rect.left + corner_radius, rect.top, rect.width - 2 * corner_radius, rect.height))


    def _init_trial_record(self):
        """内部方法：初始化试次记录文件，写入表头"""
        # 检查记录文件是否已存在，不存在则创建并写表头
        if not os.path.exists(self.record_file_path):
            with open(self.record_file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "试次轮次", 
                    "试次开始时间", "试次结束时间",
                    "情绪类型",
                    "间隔开始时间", "间隔结束时间",
                    "图片开始时间", "图片结束时间",
                    "图片路径",
                    "图片坐标", "图片尺寸",
                    "试次状态"
                ])
                writer.writeheader()
                print(f"📄 试次记录文件已创建：{self.record_file_path}")
        else:
            print(f"📄 试次记录文件已存在，将追加记录：{self.record_file_path}")

    def _write_trial_record(self, record_data):
        """内部方法：将单个试次的记录写入CSV文件"""
        try:
            # 以追加模式写入，避免覆盖历史记录
            with open(self.record_file_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "试次轮次", 
                    "试次开始时间", "试次结束时间",
                    "情绪类型",
                    "间隔开始时间", "间隔结束时间",
                    "图片开始时间", "图片结束时间",
                    "图片路径",
                    "图片坐标", "图片尺寸",
                    "试次状态"
                ])
                writer.writerow(record_data)
            print(f"✅ 试次 {record_data['试次轮次']} 记录已保存")
        except Exception as e:
            print(f"⚠️  试次 {record_data['试次轮次']} 记录写入失败：{str(e)}")

    def _load_trials_from_csv(self, csv_path):
        """内部方法：读取trials.csv中的试次参数（解决编码问题）"""
        if not os.path.exists(csv_path):
            print(f"❌ 试次文件不存在：{csv_path}")
            self._cleanup()


        trials = []
        # 尝试多种可能的编码格式
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for encoding in encodings:
            try:
                with open(csv_path, "r", encoding=encoding) as f:
                    reader = csv.DictReader(f)
                    # 尝试读取一行数据测试编码是否正确
                    next(reader)
                    # 如果读取成功，重置文件指针到开头重新读取
                    f.seek(0)
                    reader = csv.DictReader(f)

                    for row_idx, row in enumerate(reader, 2):
                        try:
                            trial = {
                            "round": int(row["轮次"]),
                            "emo_path": row["图片路径"],
                            "emo_type": row["情绪类型"].strip().lower(),  # 统一小写（可选，按需求保留）                                                                
                            "interval_dur": float(row["间隔时长（s）"]),
                            "img_dur": float(row["情绪图片时长（s）"]),
                            "blank_dur": float(row["空白屏时长（s）"])
                        }
                            trials.append(trial)
                        except KeyError as e:
                            print(f"⚠️  第{row_idx}行缺少列：{str(e)}，跳过该试次")
                        except ValueError as e:
                            print(f"⚠️  第{row_idx}行数据格式错误：{str(e)}，跳过该试次")
                print(f"✅ 成功以 {encoding} 编码读取文件")
                return trials
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"⚠️ 使用 {encoding} 编码读取时出错：{str(e)}")
                continue

        # 如果所有编码都尝试失败
        print(f"❌ 无法解析文件 {csv_path}，请检查文件编码格式")
        self._cleanup()


    def _load_target_emotion_image(self, emotion_path):
        """内部方法：加载指定情绪图片（返回图片对象+图片路径，用于记录）"""
        try:
            emotion_path = os.path.join(self.emotion_folder, emotion_path)
            image = pygame.image.load(emotion_path).convert_alpha()
            return image  # 返回（图片对象）
        except Exception as e:
            print(f"❌ 加载图片失败：{emotion_path}，错误：{str(e)}")
            return None  # 即使加载失败，也记录路径便于排查


    def _show_interval_page(self, duration, round):
        """内部方法：显示间隔页（原逻辑不变）"""
        dot_center = (self.WIDTH // 2, self.HEIGHT // 2)
        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time) >= duration*1000:
                save_path = os.path.join(self.images, f"round_{round}_中央注视图片.png")
                pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self._exit()

            self.screen.fill(self.GREY)
            # self._draw_inner_area()
            pygame.draw.circle(self.screen, self.WHITE, dot_center, self.DOT_RADIUS)
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return start_time, current_time

    def _show_single_image_page(self, image, duration, round):
        """内部方法：显示单张图片（原逻辑不变）"""
        max_width = 960
        max_height = 720
        scale_ratio = (max_width / image.get_width(), max_height / image.get_height())
        new_size = (int(image.get_width() * scale_ratio[0]), int(image.get_height() * scale_ratio[1]))
        resized_img = pygame.transform.scale(image, new_size)
        img_pos = (
            (self.WIDTH - resized_img.get_width()) // 2,
            (self.HEIGHT - resized_img.get_height()) // 2
        )

        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time)  >= duration*1000:
                save_path = os.path.join(self.images, f"round_{round}_情绪图片.png")
                pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self._exit()

            self.screen.fill(self.GREY)
            # self._draw_inner_area()
            self.screen.blit(resized_img, img_pos)

            if self.model:
                # 注视点
                _, _, bino = self.pi.get_current_gaze()
                status, gx, gy = bino
                if status:
                    gx = int(gx)
                    gy = int(gy)
                    pygame.draw.circle(self.screen, (255,255,255), (gx,gy), 20)
                    pygame.draw.circle(self.screen, (0,255,0), (gx,gy), 22, 2)

            pygame.display.flip()
            self.clock.tick(self.FPS)
        return new_size, img_pos, start_time, current_time

    def _show_blank_page(self, duration, round):
        """空白屏：（时长, 轮次），原逻辑不变"""
        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time) // 1000 >= duration:
                if round:
                    save_path = os.path.join(self.images, f"round_{round}_空白屏图片.png")
                    pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self._exit()

            self.screen.fill(self.GREY)
            # self._draw_inner_area()
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return start_time, current_time


    def _run_single_trial(self, trial_params):
        """内部方法：运行单个试次（新增记录逻辑）"""
        # 1. 解析试次参数与记录基础信息
        trial_num = trial_params["round"]
        emotion_path = trial_params["emo_path"]
        interval_duration = trial_params["interval_dur"]
        image_duration = trial_params["img_dur"]
        trial_start_time = int((datetime.now()).timestamp() * 1000)  # 精确到毫秒
        trial_status = "成功"  # 初始设为成功，失败时修改

        # print(f"\n================================ 试次 {trial_num} 开始 ================================")

        # 2. 加载图片（获取图片对象+路径，用于显示和记录）
        target_image = self._load_target_emotion_image(emotion_path)
        if target_image is None:
            trial_status = "失败（图片加载失败）"
            print(f"❌ 试次 {trial_num} 跳过")

            # 即使失败，也记录试次信息（便于排查问题）
            self._write_trial_record({
                "试次轮次": trial_num,
                "试次开始时间": trial_start_time,
                "试次结束时间": int((datetime.now()).timestamp() * 1000),
                "情绪类型": trial_params["emo_type"],
                "间隔开始时间": None,
                "间隔结束时间": None,
                "图片开始时间": None,
                "图片结束时间": None,
                "图片路径":  "无",
                "图片坐标": None,
                "图片尺寸":None,

                "试次状态": trial_status
            })
            return

        # 3. 执行试次流程
        self.pi.set_trigger(int(trial_num)*100+1)       # 触发信号（轮次相关）
        _interval_start_time, _interval_current_time = self._show_interval_page(interval_duration, trial_num)

        self.pi.set_trigger(int(trial_num)*100+2)       # 触发信号（轮次相关）
        img_size, img_pos, _img_start_time, _img_current_time = self._show_single_image_page(target_image, image_duration, trial_num)

        self.pi.set_trigger(int(trial_num)*100+3)       # 触发信号（轮次相关）
        _blank_start_time, _blank_end_time = self._show_blank_page( trial_params["blank_dur"], trial_num)

        # 4. 记录试次结果（成功/失败均记录）
        trial_end_time = int((datetime.now()).timestamp() * 1000)
        self._write_trial_record({
            "试次轮次": trial_num,
            "试次开始时间": trial_start_time,
            "试次结束时间": trial_end_time,
            "情绪类型": trial_params["emo_type"],
            "间隔开始时间": _interval_start_time,
            "间隔结束时间": _interval_current_time,
            "图片开始时间": _img_start_time,
            "图片结束时间": _img_current_time,
            "图片路径": emotion_path,  # 记录实际使用的图片完整路径
            "图片坐标": img_pos,
            "图片尺寸": img_size,
            "试次状态": trial_status
        })

    def _show_instruction_page(self):
        """显示提示语页面（鼠标可见）"""
        start_btn_rect = pygame.Rect(self.WIDTH // 2 - 100, 900, 200, 60)
        pygame.mouse.set_visible(True)

        instruction_texts = [
            "接下来屏幕上会依次出现单张图片，",
            "请您按照自己的感觉，自由观察每张图片，",
            "每张图片的观察时间约为10秒，测试大约需要5分钟。",
            "准备好之后，请按空格键 开始测试。"
        ]

        # # 提示页面保持鼠标可见
        # pygame.mouse.set_visible(True)
        running = True
        start_time = int((datetime.now()).timestamp() * 1000)
        close_time = start_time + 3 * 1000
        while running:
            for event in pygame.event.get():
                # 检测空格键或Enter键按下,仅结束当前页面
                if event.type == pygame.KEYDOWN:
                    # 空格键被按下
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        running = False
            #     if event.type == pygame.QUIT:
            #         self.quit()
            #     elif event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_ESCAPE:
            #             self.quit()
            #         elif event.key == pygame.K_SPACE:
            #             running = False

            current_time = int((datetime.now()).timestamp() * 1000)

            # if current_time >= close_time:
            #     break
            self.screen.fill(self.GREY)
            # self._draw_inner_area()
            # 绘制标题
            title_surface = self.title_font.render("单张情绪图片观看测试", True, self.BLACK)
            self.screen.blit(title_surface, title_surface.get_rect(center=(self.WIDTH // 2, 100)))

            # 绘制说明文字及图形
            y_offset = self.HEIGHT // 2 - 100*2
            line_spacing = 100
            for i, text in enumerate(instruction_texts):
                text_surface = self.text_font.render(text, True, self.BLACK)
                text_rect = text_surface.get_rect(center=(self.WIDTH // 2, y_offset))
                self.screen.blit(text_surface, text_rect)

                y_offset += line_spacing

            # 绘制开始按钮
            self._draw_rounded_rect(self.screen, self.BUTTON_BLUE, start_btn_rect, 10)
            start_text = self.font_btn.render("开始", True, self.BLACK)
            self.screen.blit(start_text, start_text.get_rect(center=start_btn_rect.center))

            if pygame.mouse.get_pressed()[0] and start_btn_rect.collidepoint(pygame.mouse.get_pos()):
                running = False

            pygame.display.flip()
            self.clock.tick(60)

    def _show_end_page(self):
        """显示实验结束页面（鼠标可见）"""
        pygame.mouse.set_visible(True)

        start_time = int(datetime.now().timestamp() * 1000)

        while int(datetime.now().timestamp() * 1000) - start_time < 2000:
            self.screen.fill(self.GREY)
            # self._draw_inner_area()
            font = self._load_font(72)
            text_surface = font.render("任务已结束，感谢参与！", True, self.BLACK)
            self.screen.blit(text_surface, text_surface.get_rect(center=self.WHITE_CIRCLE_CENTER))

            pygame.display.flip()


    def calibration_prompt(self):
        '''校准后提示姿势不同页面'''
        self.screen.fill(self.GREY)
        # self._draw_inner_area()

        tip_font = self._load_font(48)
        tip_surface = tip_font.render("请您在接下来的测试中，", True, self.BLACK)
        self.screen.blit(tip_surface,
                         tip_surface.get_rect(center=(self.WHITE_CIRCLE_CENTER[0], self.WHITE_CIRCLE_CENTER[1])))
        tip_font = self._load_font(48)
        tip_surface = tip_font.render("尽量保持身体和头部不动。", True, self.BLACK)
        self.screen.blit(tip_surface,
                         tip_surface.get_rect(center=(self.WHITE_CIRCLE_CENTER[0], self.WHITE_CIRCLE_CENTER[1] + 70)))

        pygame.display.flip()
        start_time = int((datetime.now()).timestamp() * 1000)
        close_time = start_time + 3 * 1000
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False
                # 检测空格键或Enter键按下,仅结束当前页面
                if event.type == pygame.KEYDOWN:
                    # 空格键被按下
                    if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                        waiting = False
            current_time = int((datetime.now()).timestamp() * 1000)

            if current_time >= close_time:
                break

    # 7. 外部设备交互（眼动追踪）
    def run_pi(self) -> None:
        """初始化眼动追踪设备（测试前准备）"""
        try:
            config = DefaultConfig()
            config.face_previewing = False
            self.pi = Pupilio(config)
            self.pi.create_session(session_name="deep")
            self.pi.calibration_draw(screen=self.screen, validate=False, hands_free=False)
            self.pi.start_sampling()
            print("眼动追踪设备初始化成功")
        except Exception as e:
            print(f"❌ 眼动设备初始化失败：{str(e)}")
            self._exit()  # 初始化失败时释放资源，不退出解释器

    def stop_pi(self) -> None:
        """停止眼动追踪并保存数据（测试后清理）"""
        if self.pi:  # 确保pi已初始化
            print("=== 测试结束，清理眼动设备资源 ===")
            try:
                self.pi.stop_sampling()
                self.pi.save_data(os.path.join(self.save_dir, "eyetrack.csv"))
                self.pi.release()
            except Exception as e:
                print(f"⚠️  眼动设备资源释放失败：{str(e)}")
            self.pi = None

    def _release_resources(self):
        """统一资源释放方法：仅释放Pygame和眼动设备，不退出Python解释器"""
        # 恢复鼠标显示
        if self.pygame_initialized:
            pygame.mouse.set_visible(True)
            print("🖱️  恢复鼠标显示")

        # 释放眼动设备
        self.stop_pi()

        # 释放Pygame资源（关键：只quit Pygame，不退出解释器）
        if self.pygame_initialized:
            print("🔌 释放Pygame资源")
            pygame.quit()
            self.pygame_initialized = False  # 标记Pygame已释放

        # 显示记录文件位置
        if hasattr(self, 'record_file') and os.path.exists(self.record_file):
            print(f"📄 试次记录文件位置：{os.path.abspath(self.record_file)}")

    def _exit(self):
        """内部方法：用户主动退出（ESC/窗口关闭）- 仅释放资源，不退出解释器"""
        print("🔌 检测到退出信号，开始清理资源...")
        self._release_resources()
        print("🔚 Pygame资源清理完成，Python解释器继续运行")
        # 不再调用sys.exit()，让解释器保持运行

    def _cleanup(self):
        """内部方法：文件缺失等错误退出 - 仅释放资源，不退出解释器"""
        print("🔌 检测到致命错误，开始清理资源...")
        self._release_resources()
        print("🔚 Pygame资源清理完成，Python解释器继续运行")

    def run(self):
        try:
            """外部调用方法：运行所有试次"""
            self._show_instruction_page()
            # 进入实验后隐藏鼠标
            pygame.mouse.set_visible(False)
            self._show_blank_page(1, 0)
            self.run_pi()
            self.calibration_prompt()

            print("🚀 开始执行所有试次（共 {} 个）".format(len(self.trials_list)))
            for trial in self.trials_list:
                self._run_single_trial(trial)

            print("🎉 所有试次执行完毕！")

            self._show_end_page()

            copy_file_to_folder(self.trigger_csv, self.save_dir)

            self.receive_statues = "任务完成"
            # 返回到前端
            return self.receive_statues
        except Exception as e:
            print(f"❌ 任务异常终止：{str(e)}")
            # 返回到前端
            return self.receive_statues
        finally:
            # 关键：无论是否正常执行/异常，都释放Pygame资源，不退出解释器
            self._release_resources()

def copy_file_to_folder(source_file, target_folder):
    # 检查源文件是否存在
    if not os.path.isfile(source_file):
        print(f"错误：源文件不存在 - {source_file}")
        return

    # 确保目标文件夹存在（不存在则创建，包括父目录）
    os.makedirs(target_folder, exist_ok=True)  # exist_ok=True 避免文件夹已存在时报错

    # 构建目标文件的完整路径（保留原文件名）
    file_name = os.path.basename(source_file)  # 获取源文件的文件名
    target_file = os.path.join(target_folder, file_name)  # 目标路径 = 目标文件夹 + 原文件名

    try:
        # 复制文件（保留元数据）
        shutil.copy2(source_file, target_file)
        print(f"文件复制成功：\n源文件：{source_file}\n目标路径：{target_file}")
    except PermissionError:
        print(f"错误：权限不足，无法复制文件到 {target_folder}")
    except Exception as e:
        print(f"复制失败：{str(e)}")
# ---------------------- 使用者需修改的路径配置 ----------------------
if __name__ == "__main__":
    sub_info = {
        "name": "TEST",
        "gender": "female",
        "age": "18",
        "now": datetime.now().strftime("%Y%m%d_%H%M")
    }

    test = SingleImageTrialManager(sub_info)
    result = test.run()
    print(result)
    sys.exit()