import math
import shutil

import pygame
import sys
import os
import random
import csv
from datetime import datetime  # 用于记录精确时间
from pupilio import Pupilio, DefaultConfig
from typing import List, Dict, Tuple, Optional

class EmotionTrialManager:
    WIDTH = 1920
    HEIGHT = 1080
    GRAY = (150, 150, 150)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BUTTON_BLUE = (173, 216, 230)
    CENTER = (WIDTH // 2, HEIGHT // 2)
    CORNER_RADIUS = 30
    INNER_PX = 60
    FPS = 30
    """情绪图片试次管理器：封装试次流程，支持隐藏鼠标、加载txt参数及图片记录功能"""

    def __init__(self, sub_info, model=0):
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
        
        self.model = model
        # 1. 基础配置
        self.pi = None  # 眼动设备实例，初始化时设为None
        self.pygame_initialized = False  # 标记Pygame是否已初始化

        # 情绪图片路径
        self.emotion_folder = os.path.join(self.package_root, "images")

        # 4. 加载txt试次参数
        csv_path = os.path.join(self.package_root, "TrialList", "trial_list.csv")
        self.trials_list = self._load_trials_from_csv(csv_path)

        # 3. 初始化Pygame
        pygame.init()
        self.pygame_initialized = True

        # 加载字体
        self.title_font = self._load_font(80, is_bold=True)
        self.text_font = self._load_font(36)
        self.button_font = self._load_font(32, is_bold=True)
        self.font_btn = self._load_font(30, is_bold=True)


        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("情绪图片试次流程（含图片记录功能）")
        self.clock = pygame.time.Clock()
        pygame.mouse.set_visible(False)
        print("🖱️  试次开始，隐藏鼠标显示")

        # 4. 初始化记录文件（新增）
        self.record_file = os.path.join(self.save_dir, "trial_image_records.csv")  # 记录文件路径
        self._init_record_file()  # 创建表头（若文件不存在）


        # 图片内容
        self.images = os.path.join(self.save_dir, 'images')
        os.makedirs(self.images, exist_ok=True)

        self.trigger_csv = os.path.join(self.package_root, "TrialList", "情绪自由搜索_trigger_含义.csv")

        # 返回给前端的状态
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
        font_path = os.path.join("fonts", font_filename)
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


    def _init_record_file(self):
        """初始化记录文件，写入表头（含左/中/右图片路径）"""
        if not os.path.exists(self.record_file):
            with open(self.record_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "试次轮次",
                    "试次开始时间", "试次结束时间",
                    "左情绪类型", "左图片路径", "左图片坐标",  "左图片大小",
                    "中情绪类型", "中图片路径", "中图片坐标",  "中图片大小",
                    "右情绪类型", "右图片路径", "右图片坐标",  "右图片大小",
                    "间隔开始时间", "间隔结束时间",
                    "图片开始时间", "图片结束时间",
                    "试次状态"
                ])
                writer.writeheader()
                print(f"📄 记录文件已创建：{self.record_file}")
        else:
            print(f"📄 记录文件已存在，将追加记录：{self.record_file}")


    def _write_record(self, data):
        """将单个试次的记录写入CSV文件"""
        try:
            with open(self.record_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "试次轮次",
                    "试次开始时间", "试次结束时间",
                    "左情绪类型", "左图片路径", "左图片坐标",  "左图片大小",
                    "中情绪类型", "中图片路径", "中图片坐标",  "中图片大小",
                    "右情绪类型", "右图片路径", "右图片坐标",  "右图片大小",
                    "间隔开始时间", "间隔结束时间",
                    "图片开始时间", "图片结束时间",
                    "空白屏开始时间", "空白屏结束",
                    "试次状态"
                ])
                writer.writerow(data)
            print(f"✅ 试次 {data['试次轮次']} 记录已保存")
        except Exception as e:
            print(f"⚠️  试次 {data['试次轮次']} 记录写入失败：{str(e)}")

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
                                "left_emo": row["左（情绪）"].strip().lower(),
                                "left_path": row["左（路径）"],
                                "mid_emo": row["中（情绪）"].strip().lower(),
                                "mid_path": row["中（路径）"],
                                "right_emo": row["右（情绪）"].strip().lower(),
                                "right_path": row["右（路径）"],
                                "interval_dur": float(row["间隔页时长（s）"]),
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

    def _load_single_image(self, image_path):
        """内部方法：加载单张图片（原逻辑不变）"""
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ 图片缺失：{image_path}")
                return None
            return pygame.image.load(image_path).convert_alpha()
        except Exception as e:
            print(f"⚠️ 图片加载失败：{image_path}，错误：{str(e)}")
            return None


    def _display_interval_page(self, duration, round_num):
        """修正参数顺序：（时长, 轮次），原逻辑不变"""
        dot_radius = 10
        center_pos = (self.WIDTH // 2, self.HEIGHT // 2)

        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time) // 1000 >= duration:
                save_path = os.path.join(self.images, f"round_{round_num}_中央注视图片.png")
                pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("🔚 用户按ESC退出")
                        self._exit()

            self.screen.fill(self.GRAY)
            # self._draw_inner_area()
            pygame.draw.circle(self.screen, self.WHITE, center_pos, dot_radius)
            pygame.display.flip()
            self.clock.tick(self.FPS)

        return start_time, current_time


    def _display_image_page(self, display_order, duration, round_num):
        """参数顺序保持（顺序, 时长, 轮次）"""
        resized_imgs = []
        _img_size = []

        for img in display_order:
            scale = 500/img.get_width()
            # scale = min(max_width / img.get_width(), max_height / img.get_height())
            new_size = (int(img.get_width() * scale), int(img.get_height() * scale))
            resized_imgs.append(pygame.transform.scale(img, new_size))
            _img_size.append(new_size)
            # print(new_size)

        total_img_width = sum(img.get_width() for img in resized_imgs)
        margin = (self.WIDTH - total_img_width) // 6
        current_x = margin
        img_positions = []

        for img in resized_imgs:
            y_pos = (self.HEIGHT - img.get_height()) // 2
            img_positions.append((current_x, y_pos))
            current_x += img.get_width() + margin*2

        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time) // 1000 >= duration:
                save_path = os.path.join(self.images, f"round_{round_num}_情绪图片.png")
                pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._exit()


            self.screen.fill(self.GRAY)
            # self._draw_inner_area()
            for img, pos in zip(resized_imgs, img_positions):
                self.screen.blit(img, pos)

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

        return _img_size, img_positions, start_time, current_time

    def _display_blank_page(self, duration, round_num):
        """空白屏：（时长, 轮次），原逻辑不变"""
        start_time = int((datetime.now()).timestamp() * 1000)
        running = True

        while running:
            current_time = int((datetime.now()).timestamp() * 1000)
            if (current_time - start_time) // 1000 >= duration:
                if round_num:
                    save_path = os.path.join(self.images, f"round_{round_num}_空白屏图片.png")
                    pygame.image.save(self.screen, save_path)
                running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print("🔚 用户按ESC退出")
                        self._exit()

            self.screen.fill(self.GRAY)
            # self._draw_inner_area()
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return start_time, current_time


    def _run_single_trial(self, trial_params):
        """修改：记录试次信息，尤其是左/中/右图片路径"""
        # 解析试次参数
        round = trial_params["round"]
 

        # 记录试次开始时间（精确到毫秒）
        trial_start = int((datetime.now()).timestamp() * 1000)
        trial_status = "成功"  # 默认为成功，失败时更新

        # 1. 加载左/中/右对应的图片（获取图片对象+路径）
        # 左图
        left_path = os.path.join(self.emotion_folder, trial_params["left_path"])
        left_img = self._load_single_image(left_path)
        # 中图：
        mid_path = os.path.join(self.emotion_folder, trial_params["mid_path"])
        mid_img  = self._load_single_image(mid_path)
        # 右图：
        right_path = os.path.join(self.emotion_folder, trial_params["right_path"])
        right_img  = self._load_single_image(right_path)

        # 检查图片加载状态
        if None in [left_img, mid_img, right_img]:
            trial_status = "失败（图片加载失败）"
            print(f"❌ 试次 {round} 跳过")
            # 即使失败也记录信息（便于排查）
            self._write_record({
                "试次轮次": round,
                "试次开始时间": trial_start,
                "试次结束时间": int((datetime.now()).timestamp() * 1000),
                "左情绪类型": trial_params["left_emo"],
                "左图片路径":  "加载失败",
                "左图片坐标": None,
                "左图片大小": None,
                "中情绪类型": trial_params["mid_emo"],
                "中图片路径": "加载失败",
                "中图片坐标": None,
                "中图片大小": None,
                "右情绪类型": trial_params["right_emo"],
                "右图片路径": "加载失败",
                "右图片坐标": None,
                "右图片大小": None,
                "间隔开始时间": None,
                "间隔结束时间": None,
                "图片开始时间": None,
                "图片结束时间": None,
                "试次状态": trial_status
            })
            return

        # 2. 显示间隔页（修正参数顺序：时长, 轮次）
        self.pi.set_trigger(round * 100 + 1)  # 触发信号（轮次相关）
        _interval_start_time, _interval_end_time = self._display_interval_page(trial_params["interval_dur"], round)  # 原代码参数顺序错误，已修正

        # 3. 显示图片页
        display_order = [left_img, mid_img, right_img]
        self.pi.set_trigger(round * 100 + 2)  # 触发信号（轮次相关）
        img_size, img_positions, _img_start_time, _img_end_time = self._display_image_page(display_order, trial_params["img_dur"], round)

        # 4. 显示空白屏页面
        self.pi.set_trigger(round * 100 + 3)  # 触发信号（轮次相关）
        _blank_start_time, _blank_end_time = self._display_blank_page(trial_params["interval_dur"], round)

        # 4. 记录试次结果（成功）
        trial_end = int((datetime.now()).timestamp() * 1000)
        self._write_record({
            "试次轮次": round,
            "试次开始时间": trial_start,
            "试次结束时间": trial_end,
            "左情绪类型": trial_params["left_emo"],
            "左图片路径": trial_params["left_path"],
            "左图片坐标": img_positions[0],
            "左图片大小": img_size[0],
            "中情绪类型": trial_params["mid_emo"],
            "中图片路径": trial_params["mid_path"],
            "中图片坐标": img_positions[1],
            "中图片大小": img_size[1],
            "右情绪类型": trial_params["right_emo"],
            "右图片路径": trial_params["right_path"],
            "右图片坐标": img_positions[2],
            "右图片大小": img_size[2],
            "间隔开始时间": _interval_start_time,
            "间隔结束时间": _interval_end_time,
            "图片开始时间": _img_start_time,
            "图片结束时间": _img_end_time,

            "试次状态": trial_status
        })

    def _show_instruction_page(self):
        """显示提示语页面（鼠标可见）"""
        start_btn_rect = pygame.Rect(self.WIDTH // 2 - 100, 900, 200, 60)

        instruction_texts = [
            "接下来屏幕上会依次出现三张图片，",
            "请您按照自己的感觉，自由观察每张图片，",
            "测试大约需要5分钟。",
            "准备好之后，请按空格键, 或 点击 开始测试。"
        ]

        # # 提示页面保持鼠标可见
        pygame.mouse.set_visible(True)
        running = True
        start_time = int((datetime.now()).timestamp() * 1000)
        close_time = start_time + 3 * 1000
        while running:
            for event in pygame.event.get():
                # 检测空格键或Enter键按下,仅结束当前页面
                if event.type == pygame.KEYDOWN:
                    # 空格键被按下,
                    if event.key == pygame.K_SPACE :
                        running = False
            #     elif event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_ESCAPE:
            #             self.quit()
            #         elif event.key == pygame.K_SPACE:
            #             running = False

            self.screen.fill(self.GRAY)
            # self._draw_inner_area()
            # 绘制标题
            title_surface = self.title_font.render("多张情绪图片观看测试", True, self.BLACK)
            self.screen.blit(title_surface, title_surface.get_rect(center=(self.WIDTH // 2, 100)))

            # 绘制说明文字及图形
            y_offset = 400
            line_spacing = 100
            for i, text in enumerate(instruction_texts):
                text_surface = self.text_font.render(text, True, self.BLACK)
                text_rect = text_surface.get_rect(center=(self.WIDTH // 2, y_offset))
                self.screen.blit(text_surface, text_rect)

                y_offset += line_spacing

            # 绘制开始按钮)
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
            self.screen.fill(self.GRAY)
            # self._draw_inner_area()
            font = self._load_font(72)
            text_surface = font.render("任务已结束，感谢参与！", True, self.BLACK)
            self.screen.blit(text_surface, text_surface.get_rect(center=self.CENTER))

            pygame.display.flip()

    def calibration_prompt(self):
        '''校准后提示姿势不同页面'''
        self.screen.fill(self.GRAY)
        # self._draw_inner_area()

        tip_font = self._load_font(48)
        tip_surface = tip_font.render("请您在接下来的测试中，", True, self.BLACK)
        self.screen.blit(tip_surface,
                         tip_surface.get_rect(center=(self.CENTER[0], self.CENTER[1])))
        tip_font = self._load_font(48)
        tip_surface = tip_font.render("尽量保持身体和头部不动。", True, self.BLACK)
        self.screen.blit(tip_surface,
                         tip_surface.get_rect(center=(self.CENTER[0], self.CENTER[1] + 70)))

        pygame.display.flip()
        waiting = True
        start_time = int((datetime.now()).timestamp() * 1000)
        close_time = start_time + 3 * 1000
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
            # 提示语
            self._show_instruction_page()
            # 进入实验后隐藏鼠标
            pygame.mouse.set_visible(False)
            self._display_blank_page(1, 0)
            self.run_pi()
            # 校准提示页
            self.calibration_prompt()

            """外部调用方法：运行所有试次"""
            print("🚀 开始执行所有试次")
            for trial in self.trials_list:
                self._run_single_trial(trial)

            print("🎉 所有试次执行完毕！")
            # self.stop_pi()
            self._show_end_page()

            copy_file_to_folder(self.trigger_csv, self.save_dir)
            # 返回到前端
            self.receive_statues = "任务完成"
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

    test = EmotionTrialManager(sub_info)
    result = test.run()
    print(result)
    sys.exit()