"""
main.py

Тестовий вхід у програму.
Перевіряє базовий пайплайн:
- підключення до робота;
- запуск камери;
- зчитування одного кадру;
- вивід базової інформації про кадр;
- коректне завершення роботи.
"""

from src.robot.start_up import StartUp


def main() -> None:
    start_up = StartUp(conn_type="ap")

    try:
        # Підключаємось до робота.
        start_up.connect()
        print("Підключення до робота успішне")

        # Запускаємо відеопотік.
        start_up.start_camera()
        print("Камера успішно запущена")

        # Зчитуємо один кадр.
        frame_packet = start_up.read_frame()

        # Виводимо базову інформацію про кадр.
        print(f"frame_id: {frame_packet.frame_id}")
        print(f"timestamp: {frame_packet.timestamp}")
        print(f"image is None: {frame_packet.image is None}")

        if frame_packet.image is not None:
            print(f"image shape: {frame_packet.image.shape}")

    finally:
        # Коректно завершуємо роботу з камерою та роботом.
        start_up.stop_camera()
        start_up.close()
        print("Роботу з роботом завершено коректно")


if __name__ == "__main__":
    main()