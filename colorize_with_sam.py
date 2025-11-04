"""
SAM 모델을 사용해서 이미지에서 객체를 자동 세그멘테이션하고 색상 처리하는 스크립트
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from test import CLASS_NAMES
from test.test_utils import colorize_subject


# CLASS_NAMES 정의


def get_sam_mask_auto(image_pil, processor, model, device, grid_size=5):
    """
    SAM 모델을 사용해서 이미지의 모든 객체를 자동으로 세그먼테이션
    여러 포인트를 그리드로 샘플링해서 모든 객체를 감지

    Args:
        image_pil: PIL Image
        processor: SAM processor
        model: SAM model
        device: torch device
        grid_size: 그리드 크기 (grid_size x grid_size 포인트 생성)

    Returns:
        PIL Image (binary mask)
    """
    width, height = image_pil.size

    # 그리드 포인트 생성 (이미지 전체를 커버)
    x_points = np.linspace(width * 0.15, width * 0.85, grid_size, dtype=int)
    y_points = np.linspace(height * 0.15, height * 0.85, grid_size, dtype=int)

    # 모든 마스크를 저장할 배열
    combined_mask = np.zeros((height, width), dtype=bool)

    # 각 포인트에 대해 마스크 생성
    for x in x_points:
        for y in y_points:
            input_points = [[[int(x), int(y)]]]

            # 입력 준비
            inputs = processor(
                image_pil,
                input_points=input_points,
                return_tensors="pt"
            ).to(device)

            # 추론
            with torch.no_grad():
                outputs = model(**inputs)

            # 마스크 추출
            masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )

            # 가장 좋은 마스크 선택 (첫 번째)
            mask = masks[0][0][0].numpy()  # (H, W)

            # 합치기 (OR 연산)
            combined_mask = np.logical_or(combined_mask, mask)

    # boolean to uint8
    mask_uint8 = (combined_mask * 255).astype(np.uint8)

    # PIL Image로 변환
    mask_pil = Image.fromarray(mask_uint8)

    return mask_pil


def process_images(
        input_dir="images/images",
        output_dir="outputs/colorized_sam",
        saturation=230,
        hue_range=(180, 180),
        blend_strength=1.0
):
    """
    이미지 디렉토리의 모든 이미지를 SAM으로 세그멘테이션하고 색상 처리

    Args:
        input_dir: 입력 이미지 디렉토리
        output_dir: 출력 디렉토리
        saturation: 채도 (0-255)
        hue_range: 색상 변화 범위 (x, y)
        blend_strength: 블렌딩 강도 (0.0-1.0)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 마스크 저장 디렉토리
    mask_output_path = output_path / "masks"
    mask_output_path.mkdir(exist_ok=True)

    # SAM 모델 로드
    print("Loading SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    print(f"Model loaded on {device}")

    # 이미지 파일 가져오기
    image_files = sorted(input_path.glob("*.png"))

    print(f"Found {len(image_files)} images")
    print(f"Processing images with saturation={saturation}, hue_range={hue_range}, blend_strength={blend_strength}")

    # 각 이미지 처리
    for img_path in tqdm(image_files[:1], desc="Processing images"):
        try:
            # 파일 이름에서 클래스 ID 추출
            class_id = int(img_path.stem)
            class_name = CLASS_NAMES.get(class_id, "unknown")

            # 이미지 로드
            img = Image.open(img_path).convert("RGB")

            # SAM으로 마스크 생성 (automatic - 여러 포인트 사용)
            mask = get_sam_mask_auto(img, processor, model, device, grid_size=5)

            # 색상 처리
            colorized = colorize_subject(
                img, mask,
                saturation=saturation,
                hue_range=hue_range,
                blend_strength=blend_strength
            )

            # 결과 저장
            output_file = output_path / f"{class_id}_{class_name}.png"
            colorized.save(output_file)

            # 마스크도 저장
            mask_file = mask_output_path / f"{class_id}_{class_name}_mask.png"
            mask.save(mask_file)

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue

    print(f"\nDone! Results saved to {output_path}")


if __name__ == "__main__":
    # 기본 설정으로 실행
    process_images()

    # 다른 설정으로 실행하고 싶다면:
    # process_images(saturation=255, blend_strength=0.7)
