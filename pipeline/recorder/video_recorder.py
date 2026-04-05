"""
LockVideoRecorder — Kilitlenme anlarını video olarak kaydeden modül.

Şartname gereksinimleri:
  - Sabit frame rate (20 FPS)
  - H264 codec, MP4 format
  - Kilitlenme dörtgeni kırmızı (#FF0000 = BGR (0,0,255)), kalınlık max 3px
  - Sağ üst köşede sunucu saati (milisaniye hassasiyetinde)
  - Min 640x480 çözünürlük
  - Dosya adı: [MüsabakaNo]_[TakımAdı]_[Tarih].mp4

Kullanım:
  Pipeline'ın __init__'inde:
      self.video_recorder = LockVideoRecorder(output_dir="recordings")

  Lock başladığında (lock_start_time set edildiğinde):
      self.video_recorder.start_buffering()

  Her frame'de (lock aktifken):
      self.video_recorder.add_frame(frame, bbox)

  Lock bozulduğunda (koşullar sağlanmadı):
      self.video_recorder.cancel()

  4 saniyelik lock tamamlandığında (STRIKE!):
      filepath = self.video_recorder.finalize()
"""

import cv2 as cv
import os
import time
from datetime import datetime


class LockVideoRecorder:
    def __init__(self, output_dir="recordings", fps=20,
                 team_name="Takim_Adi", musabaka_no="1"):
        """
        Args:
            output_dir: Kayıt dosyalarının kaydedileceği klasör
            fps: Sabit frame rate (şartname min 15, biz 20 kullanıyoruz)
            team_name: Takım adı (dosya isimlendirmesinde kullanılır)
            musabaka_no: Müsabaka numarası
        """
        self.output_dir = output_dir
        self.fps = fps
        self.team_name = team_name
        self.musabaka_no = musabaka_no

        # Buffer: lock süresince frame'leri burada tutuyoruz
        self._frame_buffer = []
        self._is_buffering = False
        self._lock_count = 0  # Kaç kilitlenme kaydedildi

        os.makedirs(output_dir, exist_ok=True)

    def start_buffering(self):
        """Lock başladığında çağrılır. Buffer'ı temizleyip yeni kayda başlar."""
        self._frame_buffer = []
        self._is_buffering = True
        print("[VIDEO_RECORDER] Buffering started — collecting lock frames.")

    def add_frame(self, raw_frame, bbox, lock_elapsed=0.0):
        """
        Lock aktifken her frame'de çağrılır.
        
        Args:
            raw_frame: Orijinal kamera frame'i (işlenmemiş / ham görüntü)
            bbox: [x, y, w, h] kilitlenme dörtgeni
            lock_elapsed: Kilitlenme başlangıcından bu yana geçen süre (saniye)
        """
        if not self._is_buffering:
            return
 
        # Frame'in kopyası üzerinde çizim yap (orijinali bozma)
        frame_copy = raw_frame.copy()
        fh, fw = frame_copy.shape[:2]
 
        # --- 1. Kilitlenme dörtgeni: Kırmızı (#FF0000 = BGR (0, 0, 255)) ---
        if bbox is not None:
            x, y, w, h = [int(v) for v in bbox]
            # Şartname: çizgi kalınlığı en fazla 3 piksel
            cv.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
            # Width/Height yüzdelik bilgisi (bbox üstünde)
            pct_text = f"Width:{w/fw*100:.1f}%  Height:{h/fh*100:.1f}%"
            cv.putText(frame_copy, pct_text, (x, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
 
        # --- 2. Kilitlenme süresi ---
        elapsed_text = f"LOCK: {lock_elapsed:.2f}s / 4.00s"
        cv.putText(frame_copy, elapsed_text, (10, fh - 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
        # --- 3. Saat: sağ üst köşe, milisaniye hassasiyetinde ---
        now = datetime.now()
        time_text = now.strftime("%H:%M:%S.") + f"{now.microsecond // 1000:03d}"
        
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv.getTextSize(time_text, font, font_scale, thickness)
        tx = fw - tw - 10
        ty = th + 10
        cv.rectangle(frame_copy, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv.putText(frame_copy, time_text, (tx, ty), font, font_scale, (255, 255, 255), thickness)
 
        self._frame_buffer.append(frame_copy)

    def cancel(self):
        """Lock bozulduğunda çağrılır. Buffer temizlenir, video kaydedilmez."""
        if self._is_buffering:
            frame_count = len(self._frame_buffer)
            print(f"[VIDEO_RECORDER] Lock cancelled. {frame_count} frames discarded.")
            self._frame_buffer = []
            self._is_buffering = False

    def finalize(self):
        """
        4 saniyelik lock tamamlandığında çağrılır. 
        Buffer'daki frame'leri video dosyasına yazar.
        
        Returns:
            str: Kaydedilen video dosyasının yolu, veya None (buffer boşsa)
        """
        if not self._is_buffering or len(self._frame_buffer) == 0:
            print("[VIDEO_RECORDER] No frames to save.")
            self._is_buffering = False
            return None

        self._is_buffering = False
        self._lock_count += 1

        # --- Dosya adı: [MüsabakaNo]_[TakımAdı]_[Tarih].mp4 ---
        date_str = datetime.now().strftime("%d_%m_%Y")
        filename = f"{self.musabaka_no}_{self.team_name}_{date_str}_lock{self._lock_count}.mp4"
        filepath = os.path.join(self.output_dir, filename)

        # --- VideoWriter: H264 codec, MP4 ---
        sample_frame = self._frame_buffer[0]
        fh, fw = sample_frame.shape[:2]

        # mp4v codec — OpenCV 4.5 ve ffplay ile uyumlu
        # Alternatif: 'avc1' (H264) — sisteme bağlı
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        writer = cv.VideoWriter(filepath, fourcc, self.fps, (fw, fh))

        if not writer.isOpened():
            # mp4v çalışmazsa XVID dene
            print("[VIDEO_RECORDER] mp4v failed, trying XVID...")
            fourcc = cv.VideoWriter_fourcc(*'XVID')
            filepath = filepath.replace('.mp4', '.avi')
            writer = cv.VideoWriter(filepath, fourcc, self.fps, (fw, fh))

        if not writer.isOpened():
            print("[VIDEO_RECORDER] ERROR: Could not open VideoWriter!")
            self._frame_buffer = []
            return None

        frame_count = len(self._frame_buffer)
        for frame in self._frame_buffer:
            writer.write(frame)

        writer.release()
        self._frame_buffer = []

        duration = frame_count / self.fps
        print(f"[VIDEO_RECORDER] SAVED: {filepath}")
        print(f"    Frames: {frame_count}, Duration: {duration:.2f}s, "
              f"FPS: {self.fps}, Resolution: {fw}x{fh}")
        return filepath

    @property
    def is_buffering(self):
        return self._is_buffering