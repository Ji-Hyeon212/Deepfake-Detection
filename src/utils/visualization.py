"""
Visualization Utilities
시각화 관련 유틸리티 함수들
위치: src/utils/visualization.py
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
from sklearn.metrics import confusion_matrix, roc_curve, auc


# 스타일 설정
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def visualize_detection_result(
    image: np.ndarray,
    detection: Dict,
    title: str = "Face Detection Result",
    show: bool = True,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    얼굴 검출 결과 시각화

    Args:
        image: 원본 이미지 (H, W, 3) RGB
        detection: {'bbox': [x1,y1,x2,y2], 'landmarks': (5,2), 'confidence': float}
        title: 제목
        show: plt.show() 호출 여부
        save_path: 저장 경로 (선택)

    Returns:
        시각화된 이미지
    """
    vis_img = image.copy()

    if detection is None:
        # 검출 실패
        cv2.putText(vis_img, "No face detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        # 바운딩 박스
        bbox = detection['bbox'].astype(int)
        cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     (0, 255, 0), 2)

        # 랜드마크
        landmarks = detection['landmarks'].astype(int)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        labels = ['L Eye', 'R Eye', 'Nose', 'L Mouth', 'R Mouth']

        for i, ((x, y), color, label) in enumerate(zip(landmarks, colors, labels)):
            cv2.circle(vis_img, (x, y), 3, color, -1)
            cv2.putText(vis_img, label, (x+5, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # 신뢰도
        conf_text = f"Confidence: {detection['confidence']:.3f}"
        cv2.putText(vis_img, conf_text, (bbox[0], bbox[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 시각화
    if show or save_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(vis_img)
        plt.title(title, fontsize=16)
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"저장: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    return vis_img


def visualize_alignment_comparison(
    original: np.ndarray,
    aligned: np.ndarray,
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray,
    title: str = "Alignment Comparison",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    정렬 전후 비교 시각화

    Args:
        original: 원본 이미지
        aligned: 정렬된 이미지
        src_landmarks: 원본 랜드마크
        dst_landmarks: 정렬된 랜드마크
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 원본
    axes[0].imshow(original)
    axes[0].scatter(src_landmarks[:, 0], src_landmarks[:, 1],
                   c='red', s=50, marker='x', linewidths=2)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis('off')

    # 정렬됨
    axes[1].imshow(aligned)
    axes[1].scatter(dst_landmarks[:, 0], dst_landmarks[:, 1],
                   c='green', s=50, marker='x', linewidths=2)
    axes[1].set_title("Aligned", fontsize=14)
    axes[1].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_preprocessing_pipeline(
    image: np.ndarray,
    detection: Dict,
    aligned: np.ndarray,
    quality_result: Dict,
    title: str = "Preprocessing Pipeline",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    전처리 파이프라인 전체 단계 시각화

    Args:
        image: 원본 이미지
        detection: 검출 결과
        aligned: 정렬된 얼굴
        quality_result: 품질 평가 결과
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    fig = plt.figure(figsize=(16, 6))

    # 1. 원본 이미지
    ax1 = plt.subplot(1, 4, 1)
    ax1.imshow(image)
    ax1.set_title("1. Original Image", fontsize=12)
    ax1.axis('off')

    # 2. 얼굴 검출
    ax2 = plt.subplot(1, 4, 2)
    detection_vis = image.copy()
    if detection:
        bbox = detection['bbox'].astype(int)
        cv2.rectangle(detection_vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                     (0, 255, 0), 2)
        landmarks = detection['landmarks'].astype(int)
        for x, y in landmarks:
            cv2.circle(detection_vis, (x, y), 3, (255, 0, 0), -1)
    ax2.imshow(detection_vis)
    ax2.set_title(f"2. Face Detection\nConf: {detection.get('confidence', 0):.3f}",
                 fontsize=12)
    ax2.axis('off')

    # 3. 정렬된 얼굴
    ax3 = plt.subplot(1, 4, 3)
    ax3.imshow(aligned)
    ax3.set_title("3. Aligned Face\n224x224", fontsize=12)
    ax3.axis('off')

    # 4. 품질 평가
    ax4 = plt.subplot(1, 4, 4)
    ax4.axis('off')

    # 품질 점수 텍스트
    y_pos = 0.9
    ax4.text(0.1, y_pos, "4. Quality Assessment",
            fontsize=12, weight='bold', transform=ax4.transAxes)
    y_pos -= 0.12

    valid_color = 'green' if quality_result.get('is_valid', False) else 'red'
    ax4.text(0.1, y_pos, f"Valid: {quality_result.get('is_valid', False)}",
            fontsize=10, color=valid_color, transform=ax4.transAxes)
    y_pos -= 0.1

    ax4.text(0.1, y_pos, f"Overall: {quality_result.get('overall_score', 0):.3f}",
            fontsize=10, transform=ax4.transAxes)
    y_pos -= 0.1

    # 개별 메트릭
    scores = quality_result.get('scores', {})
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            ax4.text(0.1, y_pos, f"{key}: {value:.2f}",
                    fontsize=9, transform=ax4.transAxes)
            y_pos -= 0.08

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_quality_assessment(
    image: np.ndarray,
    quality_result: Dict,
    title: str = "Quality Assessment",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    품질 평가 결과 시각화

    Args:
        image: 입력 이미지
        quality_result: 품질 평가 결과
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    fig = plt.figure(figsize=(12, 5))

    # 이미지
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)

    # 테두리 색상 (유효/무효)
    border_color = 'green' if quality_result.get('is_valid', False) else 'red'
    for spine in ax1.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(5)

    ax1.set_title("Image", fontsize=14)
    ax1.axis('off')

    # 품질 메트릭
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis('off')

    # 전체 점수
    overall = quality_result.get('overall_score', 0)
    is_valid = quality_result.get('is_valid', False)

    ax2.text(0.5, 0.95, f"Overall Score: {overall:.3f}",
            fontsize=16, weight='bold', ha='center', transform=ax2.transAxes)

    ax2.text(0.5, 0.85, f"Status: {'✅ Valid' if is_valid else '❌ Invalid'}",
            fontsize=14, ha='center', color=border_color, transform=ax2.transAxes)

    # 개별 메트릭 바 차트
    scores = quality_result.get('scores', {})
    if scores:
        metrics = list(scores.keys())
        values = list(scores.values())

        # 정규화 (0-1 범위로)
        normalized_values = []
        for key, val in zip(metrics, values):
            if key == 'blur':
                normalized_values.append(min(val / 200, 1.0))
            elif key == 'face_size':
                normalized_values.append(min(val / 300, 1.0))
            elif key == 'brightness':
                normalized_values.append(val / 255.0)
            elif key == 'contrast':
                normalized_values.append(min(val / 100, 1.0))
            else:
                normalized_values.append(val)

        y_pos = np.arange(len(metrics))

        ax_bar = fig.add_axes([0.6, 0.1, 0.35, 0.6])
        bars = ax_bar.barh(y_pos, normalized_values, color='skyblue')

        # 색상 (좋음: 녹색, 나쁨: 빨강)
        for bar, val in zip(bars, normalized_values):
            if val > 0.7:
                bar.set_color('lightgreen')
            elif val < 0.3:
                bar.set_color('lightcoral')

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(metrics)
        ax_bar.set_xlabel('Score (normalized)', fontsize=10)
        ax_bar.set_xlim(0, 1)
        ax_bar.grid(axis='x', alpha=0.3)

    # 실패 이유
    reasons = quality_result.get('reasons', [])
    if reasons:
        reason_text = "Issues:\n" + "\n".join(f"- {r}" for r in reasons)
        ax2.text(0.5, 0.2, reason_text, fontsize=9, ha='center',
                color='red', transform=ax2.transAxes)

    plt.suptitle(title, fontsize=16)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_quality_distribution(
    quality_scores: List[float],
    labels: Optional[List[int]] = None,
    title: str = "Quality Score Distribution",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    품질 점수 분포 시각화

    Args:
        quality_scores: 품질 점수 리스트
        labels: 레이블 리스트 (0=real, 1=fake)
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    plt.figure(figsize=(12, 5))

    if labels is None:
        # 전체 분포
        plt.subplot(1, 1, 1)
        plt.hist(quality_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.axvline(np.mean(quality_scores), color='red', linestyle='--',
                   label=f'Mean: {np.mean(quality_scores):.3f}')
        plt.legend()
    else:
        # Real vs Fake 분포
        quality_scores = np.array(quality_scores)
        labels = np.array(labels)

        # Real
        plt.subplot(1, 2, 1)
        real_scores = quality_scores[labels == 0]
        plt.hist(real_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Real Images (n={len(real_scores)})', fontsize=14)
        plt.axvline(real_scores.mean(), color='darkgreen', linestyle='--',
                   label=f'Mean: {real_scores.mean():.3f}')
        plt.legend()

        # Fake
        plt.subplot(1, 2, 2)
        fake_scores = quality_scores[labels == 1]
        plt.hist(fake_scores, bins=30, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Quality Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Fake Images (n={len(fake_scores)})', fontsize=14)
        plt.axvline(fake_scores.mean(), color='darkred', linestyle='--',
                   label=f'Mean: {fake_scores.mean():.3f}')
        plt.legend()

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Curves",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    학습 곡선 시각화

    Args:
        train_losses: 학습 loss
        val_losses: 검증 loss
        train_accs: 학습 정확도
        val_accs: 검증 정확도
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    epochs = range(1, len(train_losses) + 1)

    if train_accs is not None and val_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

    # Loss
    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Curves', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Accuracy
    if train_accs is not None and val_accs is not None:
        ax2.plot(epochs, train_accs, 'b-o', label='Train Acc', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-s', label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Curves', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch_samples(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    landmarks: Optional[torch.Tensor] = None,
    num_samples: int = 8,
    title: str = "Batch Samples",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    배치 샘플 시각화

    Args:
        images: (B, 3, 224, 224) 이미지 텐서 (정규화됨)
        labels: (B,) 레이블
        predictions: (B,) 예측값 (선택)
        landmarks: (B, 5, 2) 랜드마크 (선택)
        num_samples: 표시할 샘플 수
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    num_samples = min(num_samples, images.shape[0])

    # 역정규화
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # 그리드 크기 계산
    cols = min(4, num_samples)
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)

    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]

        # 이미지
        img = images_denorm[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)

        # 랜드마크
        if landmarks is not None:
            lm = landmarks[idx].cpu().numpy()
            ax.scatter(lm[:, 0], lm[:, 1], c='red', s=20, marker='x')

        # 레이블 및 예측
        label_text = "Real" if labels[idx] == 0 else "Fake"
        if predictions is not None:
            pred_text = "Real" if predictions[idx] == 0 else "Fake"
            correct = (labels[idx] == predictions[idx]).item()
            color = 'green' if correct else 'red'
            ax.set_title(f"GT: {label_text} | Pred: {pred_text}",
                        fontsize=10, color=color)
        else:
            ax.set_title(f"Label: {label_text}", fontsize=10)

        ax.axis('off')

    # 빈 서브플롯 숨기기
    for idx in range(num_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Real', 'Fake'],
    title: str = "Confusion Matrix",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    혼동 행렬 생성

    Args:
        y_true: 실제 레이블
        y_pred: 예측 레이블
        class_names: 클래스 이름
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})

    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(title, fontsize=16)

    # 정확도 계산
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'Accuracy: {accuracy:.4f}',
            ha='center', transform=plt.gca().transAxes, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve",
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    ROC 곡선 그리기

    Args:
        y_true: 실제 레이블 (0 or 1)
        y_scores: 예측 확률 (0~1)
        title: 제목
        show: 표시 여부
        save_path: 저장 경로
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"저장: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return roc_auc


def save_visualization(
    fig_or_path: Union[plt.Figure, str],
    save_path: str,
    dpi: int = 150
):
    """
    시각화 저장 헬퍼 함수

    Args:
        fig_or_path: Figure 객체 또는 현재 figure
        save_path: 저장 경로
        dpi: 해상도
    """
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(fig_or_path, str):
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        fig_or_path.savefig(save_path, dpi=dpi, bbox_inches='tight')

    print(f"시각화 저장: {save_path}")


# 테스트 코드
if __name__ == "__main__":
    """
    시각화 함수 테스트
    실행: python src/utils/visualization.py
    """
    print("Visualization 모듈 테스트\n")

    # 더미 데이터 생성
    test_image = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)

    # 얼굴처럼 보이게
    cv2.ellipse(test_image, (112, 112), (80, 100), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(test_image, (85, 95), 10, (50, 50, 50), -1)
    cv2.circle(test_image, (139, 95), 10, (50, 50, 50), -1)

    test_detection = {
        'bbox': np.array([32, 12, 192, 212]),
        'landmarks': np.array([[85, 95], [139, 95], [112, 123], [90, 160], [134, 160]]),
        'confidence': 0.95
    }

    # 테스트 1: 검출 결과
    print("1. 얼굴 검출 결과 시각화")
    visualize_detection_result(test_image, test_detection, show=False,
                              save_path="test_detection.png")

    # 테스트 2: 품질 평가
    print("2. 품질 평가 시각화")
    test_quality = {
        'is_valid': True,
        'overall_score': 0.85,
        'scores': {
            'blur': 150.0,
            'brightness': 127.0,
            'contrast': 45.0,
            'face_size': 160.0
        },
        'reasons': []
    }
    visualize_quality_assessment(test_image, test_quality, show=False,
                                save_path="test_quality.png")

    # 테스트 3: 학습 곡선
    print("3. 학습 곡선 시각화")
    train_losses = [0.8, 0.6, 0.4, 0.3, 0.25, 0.2]
    val_losses = [0.75, 0.65, 0.5, 0.4, 0.35, 0.3]
    train_accs = [60, 70, 80, 85, 88, 90]
    val_accs = [58, 68, 75, 80, 83, 85]
    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                        show=False, save_path="test_training_curves.png")

    # 테스트 4: 품질 분포
    print("4. 품질 분포 시각화")
    quality_scores = np.random.beta(2, 2, 100) * 0.5 + 0.3  # 0.3~0.8 범위
    labels = np.random.randint(0, 2, 100)
    plot_quality_distribution(quality_scores, labels, show=False,
                             save_path="test_quality_dist.png")

    # 테스트 5: 혼동 행렬
    print("5. 혼동 행렬 시각화")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    create_confusion_matrix(y_true, y_pred, show=False,
                           save_path="test_confusion_matrix.png")

    # 테스트 6: ROC 곡선
    print("6. ROC 곡선 시각화")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0] * 10)
    y_scores = np.random.rand(100)
    roc_auc = plot_roc_curve(y_true, y_scores, show=False,
                             save_path="test_roc_curve.png")
    print(f"   AUC: {roc_auc:.4f}")

    print("\n✅ 모든 시각화 테스트 완료!")
    print("\n생성된 파일:")
    print("  - test_detection.png")
    print("  - test_quality.png")
    print("  - test_training_curves.png")
    print("  - test_quality_dist.png")
    print("  - test_confusion_matrix.png")
    print("  - test_roc_curve.png")