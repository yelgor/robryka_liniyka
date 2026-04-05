"""
start_up.py

Стартовий модуль для роботи з RoboMaster через SDK.

Відповідає за:
- створення з'єднання з роботом;
- ініціалізацію камери;
- запуск відеопотоку;
- зчитування кадрів;
- упаковку кадру у FramePacket;
- коректне завершення роботи з камерою та роботом.
"""

import time
from robomaster import robot
from src.models.frame_packet import FramePacket


class StartUp:
    def __init__(self, conn_type: str = "ap"):
        """
        Ініціалізація стартового модуля.

        :param conn_type:
            Тип з'єднання з роботом.
            Для прямого Wi-Fi підключення зазвичай використовується "ap".
        """

        # Тип підключення до робота.
        self.conn_type = conn_type

        # Основний об'єкт робота з SDK.
        # Після connect() тут буде екземпляр robot.Robot().
        self.ep_robot = None

        # Модуль камери робота.
        # Після connect() тут буде доступ до camera API.
        self.ep_camera = None

        # Лічильник кадрів.
        # Збільшується кожного разу після успішного зчитування нового кадру.
        self.frame_id = 0

    def connect(self) -> None:
        """
        Створює об'єкт робота та ініціалізує з'єднання з ним.

        Після виклику цього методу:
        - робот готовий до роботи через SDK;
        - камера стає доступною через self.ep_camera.
        """

        # Створюємо об'єкт робота.
        self.ep_robot = robot.Robot()

        # Ініціалізуємо SDK-з'єднання з роботом.
        self.ep_robot.initialize(conn_type=self.conn_type)

        # Отримуємо доступ до модуля камери робота.
        self.ep_camera = self.ep_robot.camera

    def start_camera(self) -> None:
        """
        Запускає відеопотік з камери робота.

        display=False означає, що SDK сам не відкриває окреме вікно,
        а кадри будуть зчитуватись програмно через read_cv2_image().
        """

        self.ep_camera.start_video_stream(display=False)

    def read_frame(self) -> FramePacket:
        """
        Зчитує один кадр з камери та повертає його у вигляді FramePacket.

        :return:
            Об'єкт FramePacket, який містить:
            - номер кадру;
            - зображення;
            - час отримання кадру.
        """

        # Зчитуємо поточне зображення з камери.
        image = self.ep_camera.read_cv2_image()

        # Збільшуємо номер кадру.
        self.frame_id += 1

        # Формуємо структурований пакет кадру для подальшої передачі в CV-модуль.
        return FramePacket(
            frame_id=self.frame_id,
            image=image,
            timestamp=time.time()
        )

    def stop_camera(self) -> None:
        """
        Зупиняє відеопотік, якщо камера була ініціалізована.
        """

        if self.ep_camera is not None:
            self.ep_camera.stop_video_stream()

    def close(self) -> None:
        """
        Коректно закриває з'єднання з роботом.

        Цей метод потрібно викликати при завершенні роботи,
        щоб звільнити ресурси SDK.
        """

        if self.ep_robot is not None:
            self.ep_robot.close()