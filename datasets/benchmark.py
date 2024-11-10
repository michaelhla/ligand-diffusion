import time
from torch.utils.data import ConcatDataset, Subset
from pdbbind import PDBBind
from moad import MOAD
from dataloader import ProteinLigandDataLoader

def benchmark_dataloader(dataloader, num_batches=5):
    start_time = time.time()
    total_complexes = 0
    
    print("Starting dataloader benchmark...")
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            if batch is None or len(batch) == 0:
                print("Warning: Empty batch received")
                continue
                
            print(f"\nBatch {i}:")
            print(f"Batch size: {batch.num_graphs}")
            print(f"Protein atoms: {batch['protein'].pos.size(0)}")
            print(f"Ligand atoms: {batch['ligand'].pos.size(0)}")
            
            total_complexes += batch.num_graphs
            elapsed = time.time() - start_time
            complexes_per_sec = total_complexes / elapsed
            print(f"Throughput: {complexes_per_sec:.2f} complexes/sec")
                
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        
    elapsed = time.time() - start_time
    print("\nBenchmark Results:")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Total complexes: {total_complexes}")
    
if __name__ == "__main__":
    print("Creating minimal datasets for testing...")
    
    # # Create very small datasets for testing
    # pdbbind_dataset = PDBBind(
    #     root='/workspace/pdbbind/PDBBind_processed',
    #     cache_path=None,  # Disable caching for streaming test
    #     split='train',
    #     num_workers=4,
    #     precompute=False
    # )
    
    moad_dataset = MOAD(
        root='/workspace/dockgen/BindingMOAD_2020_processed',
        cache_path=None,  # Disable caching for streaming test
        split='train',
        num_workers=4,
        precompute=False
    )
    # Add debug prints for dataset
    # print(f"\nPDBBind dataset size: {len(pdbbind_dataset)}")
    # print("Testing first item:")
    # first_item = pdbbind_dataset[0]
    # if first_item is not None:
    #     print(f"First item keys: {first_item.keys}")
    # else:
    #     print("First item is None!")
    
    # # Test with very small batch size and fewer workers for testing
    # print("\nTesting PDBBind streaming:")
    # pdbbind_loader = ProteinLigandDataLoader(
    #     pdbbind_dataset,
    #     batch_size=32,  # Small batch size
    #     shuffle=True,
    #     num_workers=2,  # Fewer workers
    # )
    # benchmark_dataloader(pdbbind_loader)
    
    print("\nTesting MOAD streaming:")
    moad_loader = ProteinLigandDataLoader(
        moad_dataset,
        batch_size=64,  # Small batch size
        shuffle=True,
        num_workers=2  # Fewer workers
    )
    benchmark_dataloader(moad_loader)
