from ultralytics import YOLO

def main():
    base_model = YOLO("runs/detect/train-8/weights/best.pt")
    fine_model = YOLO("runs/detect/train-final-5/weights/best.pt")

    data_path = "dataset_kz_test/data.yaml"

    print("\n========== BASE MODEL ==========")

    base_results = base_model.val(
        data=data_path,
        split="test",
        imgsz=320,
        batch=16,
        workers=0
    )

    print("\n========== FINE-TUNED MODEL ==========")

    fine_results = fine_model.val(
        data=data_path,
        split="test",
        imgsz=320,
        batch=16,
        workers=0
    )

    print("\n========== COMPARISON ==========")

    print("\nBASE MODEL")
    print(f"Precision:  {base_results.box.mp:.4f}")
    print(f"Recall:     {base_results.box.mr:.4f}")
    print(f"mAP@0.5:    {base_results.box.map50:.4f}")
    print(f"mAP50-95:   {base_results.box.map:.4f}")

    print("\nFINE-TUNED MODEL")
    print(f"Precision:  {fine_results.box.mp:.4f}")
    print(f"Recall:     {fine_results.box.mr:.4f}")
    print(f"mAP@0.5:    {fine_results.box.map50:.4f}")
    print(f"mAP50-95:   {fine_results.box.map:.4f}")

    print("\n========== mAP50-95 BY CLASS ==========")

    for i, class_name in base_results.names.items():
        print(f"\nClass: {class_name}")
        print(f"BASE: {base_results.box.maps[i]:.4f}")
        print(f"FINE: {fine_results.box.maps[i]:.4f}")


if __name__ == "__main__":
    main()