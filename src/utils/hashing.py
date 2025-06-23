"""
utils/hashing.py

This utility module provides a deterministic way to create a "fingerprint"
of a local region's state. It uses a cryptographic hashing algorithm (SHA-256)
to ensure that the same local configuration of tags always produces the
same unique hash.
"""
import hashlib
import numpy as np

def hash_cell_state(cell_tags):
    """
    Computes a SHA-256 hash for the state of a single grid cell.

    To ensure the hash is deterministic and not dependent on the order of nodes,
    the tags (which are in a NumPy array) are converted directly into a byte
    string. This serves as a canonical representation.

    Args:
        cell_tags (np.ndarray): A NumPy array of integer tags for all nodes
                                within a specific grid cell.

    Returns:
        str: A hexadecimal string representing the SHA-256 hash of the cell's
             state. Returns None if the input is empty.
    """
    if cell_tags.size == 0:
        return None

    # Convert the NumPy array directly to bytes. This is fast and deterministic.
    byte_representation = cell_tags.tobytes()

    # Create the hash object and update it with the byte representation.
    hasher = hashlib.sha256()
    hasher.update(byte_representation)

    # Return the hexadecimal digest of the hash.
    return hasher.hexdigest()


# --- Standalone Test Block ---
if __name__ == '__main__':
    print("--- Running Standalone Test for utils/hashing.py ---")

    # Create some sample cell states
    state1 = np.array([5, 8, 0, 12, 3], dtype=int)
    state2 = np.array([5, 8, 0, 12, 3], dtype=int)  # Identical to state1
    state3 = np.array([5, 8, 1, 12, 3], dtype=int)  # Different from state1
    state4 = np.array([3, 12, 0, 8, 5], dtype=int)  # Different order

    # Compute hashes
    hash1 = hash_cell_state(state1)
    hash2 = hash_cell_state(state2)
    hash3 = hash_cell_state(state3)
    hash4 = hash_cell_state(state4)

    print(f"\nState 1: {state1}")
    print(f"Hash 1:  {hash1}")

    print(f"\nState 2: {state2}")
    print(f"Hash 2:  {hash2}")

    print(f"\nState 3: {state3}")
    print(f"Hash 3:  {hash3}")
    
    print(f"\nState 4: {state4} (different order)")
    print(f"Hash 4:  {hash4}")

    print("\n--- Verification Checks ---")
    if hash1 and hash1 == hash2:
        print("Verification PASSED: Identical states produce identical hashes.")
    else:
        print("Verification FAILED: Identical states produced different hashes.")

    if hash1 and hash1 != hash3:
        print("Verification PASSED: Different states produce different hashes.")
    else:
        print("Verification FAILED: Different states produced the same hash.")
        
    if hash1 and hash1 != hash4:
        print("Verification PASSED: Different order of elements produces different hashes.")
    else:
        print("Verification FAILED: Order of elements did not affect the hash.")
    
    print("\n--- Test Complete ---")

