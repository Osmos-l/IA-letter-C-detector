import time

def benchmark(func, *args, **kwargs):
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time() 

    elapsed_time = end_time - start_time
    print(f"Temps d'exécution: {elapsed_time:.6f} secondes.")
    print("\n")
    