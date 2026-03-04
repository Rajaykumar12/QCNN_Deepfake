# Dataset Split Calculator for 10,000 Image Constraint

## Default Split Configuration
TOTAL_IMAGES = 10000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10  
TEST_RATIO = 0.10

def calculate_dataset_splits(total_images=10000):
    """
    Calculate dataset splits for the 10,000 image constraint.
    
    Args:
        total_images: Total number of images available (including train/val/test)
    
    Returns:
        dict: Split allocations
    """
    train_images = int(total_images * TRAIN_RATIO)
    val_images = int(total_images * VAL_RATIO)
    test_images = total_images - train_images - val_images  # Remainder to test
    
    # Per class (assuming balanced)
    train_per_class = train_images // 2
    val_per_class = val_images // 2
    test_per_class = test_images // 2
    
    splits = {
        'total_images': total_images,
        'train': {
            'total': train_images,
            'per_class': train_per_class,
            'fake': train_per_class,
            'real': train_per_class
        },
        'validation': {
            'total': val_images,
            'per_class': val_per_class,
            'fake': val_per_class,
            'real': val_per_class
        },
        'test': {
            'total': test_images,
            'per_class': test_per_class,
            'fake': test_per_class,
            'real': test_per_class
        }
    }
    
    return splits

if __name__ == "__main__":
    print("📊 Dataset Split Calculator for 10,000 Image Constraint")
    print("=" * 60)
    
    splits = calculate_dataset_splits()
    
    print(f"\n🎯 Target Dataset Composition:")
    print(f"  Total Images: {splits['total_images']:,}")
    print(f"  Classes: 2 (fake, real)")
    print(f"  Balance: 5,000 fake + 5,000 real")
    
    print(f"\n📈 Recommended Splits:")
    print(f"  Training:   {splits['train']['total']:,} images ({TRAIN_RATIO:.0%})")
    print(f"    ├─ Fake:  {splits['train']['fake']:,}")
    print(f"    └─ Real:  {splits['train']['real']:,}")
    
    print(f"\n  Validation: {splits['validation']['total']:,} images ({VAL_RATIO:.0%})")
    print(f"    ├─ Fake:  {splits['validation']['fake']:,}")
    print(f"    └─ Real:  {splits['validation']['real']:,}")
    
    print(f"\n  Testing:    {splits['test']['total']:,} images ({TEST_RATIO:.0%})")
    print(f"    ├─ Fake:  {splits['test']['fake']:,}")
    print(f"    └─ Real:  {splits['test']['real']:,}")
    
    print(f"\n✅ Total Verification: {splits['train']['total'] + splits['validation']['total'] + splits['test']['total']:,} images")
    
    print(f"\n📝 Usage Notes:")
    print(f"  • This split is automatically applied by training scripts")
    print(f"  • Use --max_samples 10000 to enforce this limit")
    print(f"  • The system maintains class balance across all splits")
    print(f"  • Quantum preprocessing works efficiently with this size")
    
    print(f"\n🔧 Command Line Example:")
    print(f"  python train_cnn.py --dataset_dir ./Dataset --max_samples 10000")