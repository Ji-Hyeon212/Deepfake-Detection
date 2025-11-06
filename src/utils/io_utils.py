"""
I/O Utilities
파일 입출력 관련 유틸리티
위치: src/utils/io_utils.py
"""

import json
import yaml
import pickle
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import shutil


def load_json(file_path: str) -> Dict:
    """
    JSON 파일 로드

    Args:
        file_path: JSON 파일 경로

    Returns:
        딕셔너리
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict, file_path: str, indent: int = 2):
    """
    JSON 파일 저장

    Args:
        data: 저장할 데이터
        file_path: 저장 경로
        indent: 들여쓰기
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    print(f"JSON 저장: {file_path}")


def load_yaml(file_path: str) -> Dict:
    """
    YAML 파일 로드

    Args:
        file_path: YAML 파일 경로

    Returns:
        딕셔너리
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: str):
    """
    YAML 파일 저장

    Args:
        data: 저장할 데이터
        file_path: 저장 경로
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML 저장: {file_path}")


def save_checkpoint(
        state: Dict[str, Any],
        save_path: str,
        is_best: bool = False
):
    """
    모델 체크포인트 저장

    Args:
        state: 저장할 상태 (model, optimizer, epoch, etc.)
        save_path: 저장 경로
        is_best: 최고 성능 모델 여부
    """
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(state, save_path)
    print(f"체크포인트 저장: {save_path}")

    if is_best:
        best_path = output_path.parent / "best_model.pth"
        shutil.copyfile(save_path, best_path)
        print(f"최고 모델 저장: {best_path}")


def load_checkpoint(
        checkpoint_path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cuda'
) -> Dict[str, Any]:
    """
    체크포인트 로드

    Args:
        checkpoint_path: 체크포인트 경로
        model: 모델 (선택)
        optimizer: 옵티마이저 (선택)
        device: 디바이스

    Returns:
        체크포인트 딕셔너리
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 가중치 로드: {checkpoint_path}")

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"옵티마이저 상태 로드: {checkpoint_path}")

    return checkpoint


def save_pickle(data: Any, file_path: str):
    """
    Pickle 파일 저장

    Args:
        data: 저장할 데이터
        file_path: 저장 경로
    """
    output_path = Path(file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Pickle 저장: {file_path}")


def load_pickle(file_path: str) -> Any:
    """
    Pickle 파일 로드

    Args:
        file_path: Pickle 파일 경로

    Returns:
        로드된 데이터
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str):
    """
    디렉토리 존재 확인 및 생성

    Args:
        directory: 디렉토리 경로
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """
    프로젝트 루트 디렉토리 반환

    Returns:
        프로젝트 루트 Path 객체
    """
    return Path(__file__).parent.parent.parent


# 테스트 코드
if __name__ == "__main__":
    print("I/O Utils 테스트\n")

    # 테스트 데이터
    test_data = {
        'model': 'EfficientNet-B4',
        'epoch': 10,
        'accuracy': 0.95,
        'settings': {
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }

    # JSON 테스트
    print("1. JSON 저장/로드 테스트")
    save_json(test_data, "test_output/test.json")
    loaded = load_json("test_output/test.json")
    assert loaded == test_data
    print("   ✅ 성공\n")

    # YAML 테스트
    print("2. YAML 저장/로드 테스트")
    save_yaml(test_data, "test_output/test.yaml")
    loaded = load_yaml("test_output/test.yaml")
    assert loaded == test_data
    print("   ✅ 성공\n")

    # Pickle 테스트
    print("3. Pickle 저장/로드 테스트")
    save_pickle(test_data, "test_output/test.pkl")
    loaded = load_pickle("test_output/test.pkl")
    assert loaded == test_data
    print("   ✅ 성공\n")

    print("✅ 모든 I/O Utils 테스트 완료!")