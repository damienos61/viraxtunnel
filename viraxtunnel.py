#!/usr/bin/env python3
"""
VIRAXTUNNEL — ULTIMATE BATTLE-READY V3 (hardened & corrected)
The channel is dead. Only information survives in the chaos.
Corrected and operational - Secure Edition
"""
import argparse
import base64
import hashlib
import json
import os
import random
import secrets
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Tuple

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---------------------------
# Utilities
# ---------------------------
def _secure_write(path: Path, data: bytes, mode: str = "wb"):
    """Write file atomically and set restrictive permissions when possible."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    tmp.replace(path)
    try:
        # chmod only meaningful on POSIX
        path.chmod(0o600)
    except Exception:
        pass

def _now_ts() -> int:
    return int(time.time())

# ============================================================
# 1. GHOST IDENTITY (FIXED)
# ============================================================
class GhostIdentity:
    def __init__(self, identity_name: str = "default", password: Optional[str] = None, generate_new: bool = False):
        self.identity_name = identity_name
        self.sign_priv = None  # type: ignore
        self.enc_priv = None  # type: ignore

        sign_path = Path(f"{identity_name}.sign")
        enc_path = Path(f"{identity_name}.enc")

        if not generate_new and sign_path.exists() and enc_path.exists():
            self._load(sign_path, enc_path, password)
        else:
            print(f"[+] Generating new keys for identity '{identity_name}'")
            self.sign_priv = ed25519.Ed25519PrivateKey.generate()
            # RSA 3072 for a good compromise
            self.enc_priv = rsa.generate_private_key(
                public_exponent=65537,
                key_size=3072
            )
            self._save(sign_path, enc_path, password)

        self.sign_pub = self.sign_priv.public_key()
        self.enc_pub = self.enc_priv.public_key()

    def _load(self, sign_path: Path, enc_path: Path, password: Optional[str] = None):
        """Load keys from the filesystem"""
        pw = password.encode() if password else None
        try:
            with open(sign_path, "rb") as f:
                self.sign_priv = serialization.load_pem_private_key(f.read(), password=pw)
            with open(enc_path, "rb") as f:
                self.enc_priv = serialization.load_pem_private_key(f.read(), password=pw)
            print(f"[+] Identity '{self.identity_name}' loaded")
        except Exception as e:
            print(f"[-] Key load error: {e}")
            raise

    def _save(self, sign_path: Path, enc_path: Path, password: Optional[str] = None):
        """Save keys to the filesystem with restrictive permissions"""
        encryption = serialization.BestAvailableEncryption(password.encode()) if password else serialization.NoEncryption()

        sign_bytes = self.sign_priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )

        enc_bytes = self.enc_priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption
        )

        _secure_write(sign_path, sign_bytes)
        _secure_write(enc_path, enc_bytes)
        print(f"[+] Keys saved: {sign_path}, {enc_path}")

    def get_public_bundle(self) -> str:
        """Return public keys as JSON for sharing (compatible)"""
        # Ed25519 public key as raw bytes (base64)
        sign_pub_bytes = self.sign_pub.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        )

        # RSA public key as PEM bytes (base64 encoded to ensure JSON safety)
        enc_pub_pem = self.enc_pub.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return json.dumps({
            "identity": self.identity_name,
            "sign": base64.b64encode(sign_pub_bytes).decode(),
            "enc": base64.b64encode(enc_pub_pem).decode(),
            "timestamp": _now_ts()
        }, indent=2, ensure_ascii=False)

    @staticmethod
    def load_public_bundle(bundle_str: str):
        """Load a public bundle from JSON (compatible with get_public_bundle)"""
        bundle = json.loads(bundle_str)
        sign_pub_bytes = base64.b64decode(bundle["sign"])
        enc_pub_bytes = base64.b64decode(bundle["enc"])

        sign_pub = ed25519.Ed25519PublicKey.from_public_bytes(sign_pub_bytes)
        enc_pub = serialization.load_pem_public_key(enc_pub_bytes)

        return {
            "identity": bundle.get("identity", "unknown"),
            "sign_pub": sign_pub,
            "enc_pub": enc_pub
        }

# ============================================================
# 2. IMPROVED STEGANOGRAPHIC CARRIER
# ============================================================
class ViraxCarrier:
    def __init__(self, seed: Optional[int] = None):
        """Initialize with a seed for reproducibility.
        If seed is None, use random.SystemRandom (cryptographically secure).
        """
        self.palette = ['\u200b', '\u200c', '\u200d', '\u200e', '\u200f', '\ufeff']
        if seed is not None:
            self.rng = random.Random(seed)  # reproducible (not cryptographically secure)
        else:
            self.rng = random.SystemRandom()  # cryptographically secure RNG with same API

    def embed(self, data: str, cover: str) -> str:
        """Embed data into a cover text"""
        # Add a CRC32 checksum for verification
        crc = zlib.crc32(data.encode()) & 0xFFFFFFFF
        data_with_crc = f"{crc:08x}:{data}"

        # Base64 encode for reliability
        encoded_data = base64.b64encode(data_with_crc.encode()).decode()

        # Convert to binary
        binary = ''.join(format(ord(c), '08b') for c in encoded_data)

        # Choose two distinct control characters
        c0, c1 = self.rng.sample(self.palette, 2)

        # Start marker
        marker = '\u200b\u200c\u200d'

        # Stego body: marker + c0 + c1 + bits
        encoded = marker + c0 + c1 + ''.join(c0 if bit == '0' else c1 for bit in binary)

        # Insert into cover text: insert into words to limit alteration
        words = cover.split(' ')
        if not words:
            words = [cover]

        num_insertions = max(1, min(3, len(words) // 5))
        for _ in range(num_insertions):
            pos = self.rng.randint(0, len(words) - 1)
            insertion_point = self.rng.randint(0, max(0, len(words[pos])))
            words[pos] = words[pos][:insertion_point] + encoded + words[pos][insertion_point:]

        return ' '.join(words)

    def extract(self, text: str) -> Optional[str]:
        """Extract hidden data"""
        marker = '\u200b\u200c\u200d'
        idx = text.find(marker)
        if idx == -1:
            return None
        start_idx = idx + len(marker)
        if start_idx >= len(text) - 2:
            return None

        # Collect stego characters while they belong to the palette
        steg_chars = []
        i = start_idx
        while i < len(text) and text[i] in self.palette:
            steg_chars.append(text[i])
            i += 1

        # Need at least two (c0,c1) + some bits
        if len(steg_chars) < 8:
            return None

        c0, c1 = steg_chars[0], steg_chars[1]
        bits = steg_chars[2:]
        binary_str = ''.join('0' if ch == c0 else '1' for ch in bits)

        try:
            # Adjust length to multiple of 8
            if len(binary_str) % 8 != 0:
                binary_str = binary_str[:-(len(binary_str) % 8)]

            bytes_chars = bytearray()
            for j in range(0, len(binary_str), 8):
                byte = binary_str[j:j+8]
                bytes_chars.append(int(byte, 2))

            encoded = bytes(bytes_chars).decode()
            decoded = base64.b64decode(encoded).decode()

            crc_str, data = decoded.split(':', 1)
            expected_crc = int(crc_str, 16)
            actual_crc = zlib.crc32(data.encode()) & 0xFFFFFFFF

            if expected_crc == actual_crc:
                return data
            else:
                # CRC mismatch
                return None
        except Exception:
            return None

# ============================================================
# 3. ROBUST FRAGMENTATION SYSTEM
# ============================================================
class ViraxEntropy:
    @staticmethod
    def fragment(data_packet: str, num_fragments: int = 3, redundancy: int = 2) -> List[str]:
        """Fragment data with redundancy and checksums"""
        packet_hash = hashlib.sha256(data_packet.encode()).hexdigest()[:16]

        # Split into fragments (contiguous distribution)
        chunk_size = (len(data_packet) + num_fragments - 1) // num_fragments
        fragments: List[str] = []

        for i in range(num_fragments):
            start = i * chunk_size
            end = min(start + chunk_size, len(data_packet))
            chunk = data_packet[start:end]
            fragment = {
                "id": packet_hash,
                "index": i,
                "total": num_fragments,
                "data": chunk,
                "checksum": hashlib.sha1(chunk.encode()).hexdigest()[:8]
            }
            fragments.append(json.dumps(fragment))

        # Redundancy and shuffle
        redundant_fragments: List[str] = []
        for frag in fragments:
            redundant_fragments.extend([frag] * redundancy)
        random.shuffle(redundant_fragments)
        return redundant_fragments

    @staticmethod
    def reconstruct(fragments: List[str]) -> str:
        """Reconstruct data from fragments (majority vote)"""
        from collections import Counter, defaultdict

        # packet_id -> {"total": int or None, "parts": defaultdict(list)}
        packets = defaultdict(lambda: {"total": None, "parts": defaultdict(list)})

        for frag_str in fragments:
            try:
                frag = json.loads(frag_str)
                packet_id = frag["id"]
                index = frag["index"]
                data = frag["data"]
                checksum = frag["checksum"]

                if hashlib.sha1(data.encode()).hexdigest()[:8] != checksum:
                    # corrupted fragment
                    continue

                if packets[packet_id]["total"] is None:
                    packets[packet_id]["total"] = frag.get("total")
                packets[packet_id]["parts"][index].append(data)
            except Exception:
                # Ignore malformed fragments
                continue

        # For each packet, try reconstruction
        for packet_id, info in packets.items():
            total = info["total"]
            parts = info["parts"]
            if not total:
                continue

            reconstructed = {}
            for i in range(total):
                if i in parts and parts[i]:
                    counter = Counter(parts[i])
                    most_common = counter.most_common(1)
                    if most_common:
                        reconstructed[i] = most_common[0][0]

            if len(reconstructed) == total:
                return ''.join(reconstructed[i] for i in range(total))

        raise ValueError("Unable to reconstruct the packet")

# ============================================================
# 4. IMPROVED & SECURE CRYPTOGRAPHIC PROTOCOL
# ============================================================
class ViraxProtocol:
    @staticmethod
    def encrypt_message(
        sender_priv: ed25519.Ed25519PrivateKey,
        receiver_pub: rsa.RSAPublicKey,
        message: str,
        ttl: int = 3600
    ) -> str:
        """Encrypt a message for a specific recipient.
        Format: [4-bytes packet_len][packet][signature]
        where packet = wrapped_key || iv || ciphertext_with_tag
        """
        timestamp = _now_ts()
        payload = f"{timestamp}:{ttl}:{message}".encode()

        # Compression (speed/ratio compromise)
        compressed = zlib.compress(payload, level=6)

        # AES-256-GCM via AESGCM (high-level)
        aes_key = secrets.token_bytes(32)
        aesgcm = AESGCM(aes_key)
        iv = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(iv, compressed, None)  # includes tag at end

        # Wrap AES key with RSA-OAEP
        wrapped_key = receiver_pub.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        packet = wrapped_key + iv + ciphertext

        # Ed25519 signature
        signature = sender_priv.sign(packet)

        # Prefix packet length (big-endian 4 bytes) for safe parsing
        full_packet = struct.pack(">I", len(packet)) + packet + signature
        return base64.b64encode(full_packet).decode()

    @staticmethod
    def decrypt_message(
        receiver_priv: rsa.RSAPrivateKey,
        sender_pub: ed25519.Ed25519PublicKey,
        encrypted_message: str
    ) -> Tuple[str, int]:
        """Decrypt a received message and verify signature + TTL"""
        try:
            full_packet = base64.b64decode(encrypted_message)
            if len(full_packet) < 4:
                raise ValueError("Packet too short")

            packet_len = struct.unpack(">I", full_packet[:4])[0]
            if len(full_packet) < 4 + packet_len:
                raise ValueError("Truncated packet")

            packet = full_packet[4:4 + packet_len]
            signature = full_packet[4 + packet_len:]

            # Verify signature (may raise InvalidSignature)
            sender_pub.verify(signature, packet)

            # Compute wrapped_key length based on receiver RSA key
            wrapped_key_len = receiver_priv.key_size // 8
            if len(packet) < wrapped_key_len + 12 + 16:
                raise ValueError("Packet malformed (too short for expected components)")

            wrapped_key = packet[:wrapped_key_len]
            iv = packet[wrapped_key_len:wrapped_key_len + 12]
            ciphertext = packet[wrapped_key_len + 12:]  # contains tag

            # Decrypt AES key
            aes_key = receiver_priv.decrypt(
                wrapped_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )

            # AES-GCM decryption
            aesgcm = AESGCM(aes_key)
            compressed = aesgcm.decrypt(iv, ciphertext, None)

            payload = zlib.decompress(compressed).decode()
            timestamp_str, ttl_str, message = payload.split(":", 2)
            timestamp = int(timestamp_str)
            ttl = int(ttl_str)
            if _now_ts() - timestamp > ttl:
                raise ValueError(f"Message expired (TTL: {ttl}s)")

            return message, timestamp
        except InvalidSignature:
            raise ValueError("Invalid signature - message corrupted or forged")
        except InvalidTag:
            raise ValueError("Authentication failed - corrupted data")
        except Exception as e:
            raise ValueError(f"Decryption error: {e}")

# ============================================================
# 5. UTILITY FUNCTIONS
# ============================================================
def generate_cover_texts(count: int) -> List[str]:
    """Generate random cover texts"""
    base = [
        "Coffee is ready in the kitchen, don't forget.",
        "The meeting has been moved to tomorrow morning.",
        "Could you check the server logs tonight?",
        "The deployment completed successfully.",
        "I updated the technical documentation.",
        "The quarterly report is being finalized.",
        "Metrics show a significant improvement.",
        "Remember to backup data before maintenance.",
        "The new release will be published next week.",
        "All integration tests have passed.",
    ]
    covers = list(base)
    rng = random.Random(_now_ts())
    while len(covers) < count:
        covers.append(f"Internal note {rng.randint(1000, 9999)}: Verification scheduled for {rng.choice(['Monday','Tuesday','Wednesday','Thursday','Friday'])}.")
    return covers[:count]

def save_transmission(transmission: List[str], filename: str = "transmission.json"):
    """Save the transmission to a file atomically"""
    data = {
        "timestamp": _now_ts(),
        "count": len(transmission),
        "transmission": transmission
    }
    path = Path(filename)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    print(f"[+] Transmission saved to {filename}")

# ============================================================
# 6. MAIN INTERFACE
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="VIRAXTUNNEL V3 - Furtive and resilient communication system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new identity
  python viraxtunnel.py --new-identity alice --password mypassword

  # Get the public bundle
  python viraxtunnel.py --identity alice --get-public

  # Send a message
  python viraxtunnel.py --identity alice --receiver-pub bob.pub.json --message "Secret message"

  # Receive a message
  python viraxtunnel.py --identity bob --decrypt transmission.json --sender-pub alice.pub.json
        """
    )

    parser.add_argument('--new-identity', type=str, help='Create a new identity')
    parser.add_argument('--identity', type=str, help='Use an existing identity')
    parser.add_argument('--password', type=str, help='Password for the keys')

    parser.add_argument('--message', type=str, help='Message to send')
    parser.add_argument('--receiver-pub', type=str, help="Recipient's public bundle file")
    parser.add_argument('--output', type=str, default='transmission.json', help='Output file')
    parser.add_argument('--fragments', type=int, default=5, help='Number of fragments')
    parser.add_argument('--redundancy', type=int, default=2, help='Redundancy factor')

    parser.add_argument('--decrypt', type=str, help='File to decrypt')
    parser.add_argument('--sender-pub', type=str, help="Sender's public bundle file")

    parser.add_argument('--get-public', action='store_true', help='Display the public bundle')
    parser.add_argument('--test', action='store_true', help='Run a full test')

    args = parser.parse_args()

    # Mode 1: Create identity
    if args.new_identity:
        print(f"[+] Creating identity '{args.new_identity}'")
        identity = GhostIdentity(args.new_identity, args.password, generate_new=True)
        print("[+] Identity created successfully")
        print("[+] Public bundle:\n" + identity.get_public_bundle())
        return

    # Mode 2: Display public bundle
    if args.identity and args.get_public:
        identity = GhostIdentity(args.identity, args.password)
        print(identity.get_public_bundle())
        return

    # Mode 3: Full test
    if args.test:
        print("[+] Running full test...")
        try:
            alice = GhostIdentity("test_alice", "test123", generate_new=True)
            bob = GhostIdentity("test_bob", "test123", generate_new=True)

            alice_bundle = alice.get_public_bundle()
            bob_bundle = bob.get_public_bundle()

            alice_pub_info = GhostIdentity.load_public_bundle(alice_bundle)
            bob_pub_info = GhostIdentity.load_public_bundle(bob_bundle)

            message = "This is a test message for ViraxTunnel V3"
            encrypted = ViraxProtocol.encrypt_message(
                alice.sign_priv,
                bob_pub_info["enc_pub"],
                message
            )

            fragments = ViraxEntropy.fragment(encrypted, num_fragments=3, redundancy=2)
            carrier = ViraxCarrier()
            covers = generate_cover_texts(len(fragments))
            transmission = []
            for i, frag in enumerate(fragments):
                stego_text = carrier.embed(frag, covers[i])
                transmission.append(stego_text)

            # Extraction
            extracted_frags = []
            for text in transmission:
                extracted = carrier.extract(text)
                if extracted:
                    extracted_frags.append(extracted)

            reconstructed = ViraxEntropy.reconstruct(extracted_frags)
            decrypted, timestamp = ViraxProtocol.decrypt_message(
                bob.enc_priv,
                alice_pub_info["sign_pub"],
                reconstructed
            )

            print("\n[✓] TEST SUCCESS!")
            print(f"Original message: {message}")
            print(f"Decrypted message: {decrypted}")
            print(f"Timestamp: {timestamp}")
        finally:
            # Cleanup test keys if present
            for p in ("test_alice.sign", "test_alice.enc", "test_bob.sign", "test_bob.enc"):
                try:
                    Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
        return

    # Mode 4: Send message
    if args.identity and args.message and args.receiver_pub:
        print("[+] SEND mode activated")
        sender = GhostIdentity(args.identity, args.password)
        with open(args.receiver_pub, "r", encoding="utf-8") as f:
            receiver_info = GhostIdentity.load_public_bundle(f.read())

        print(f"[+] Encrypting for {receiver_info['identity']}...")
        encrypted = ViraxProtocol.encrypt_message(
            sender.sign_priv,
            receiver_info["enc_pub"],
            args.message
        )

        print(f"[+] Fragmenting into {args.fragments} parts (redundancy {args.redundancy})...")
        fragments = ViraxEntropy.fragment(encrypted, num_fragments=args.fragments, redundancy=args.redundancy)

        carrier = ViraxCarrier()
        covers = generate_cover_texts(len(fragments))
        transmission = []
        for i, frag in enumerate(fragments):
            stego_text = carrier.embed(frag, covers[i])
            transmission.append(stego_text)
            print(f"[+] Fragment {i} inserted into cover.")

        save_transmission(transmission, args.output)
        print(f"\n[✓] SEND COMPLETE — {len(fragments)} fragments generated.")
        return

    # Mode 5: Receive message
    if args.identity and args.decrypt and args.sender_pub:
        print("[+] RECEIVE mode activated")
        receiver = GhostIdentity(args.identity, args.password)
        with open(args.sender_pub, "r", encoding="utf-8") as f:
            sender_info = GhostIdentity.load_public_bundle(f.read())

        with open(args.decrypt, "r", encoding="utf-8") as f:
            data = json.load(f)
            transmission = data.get("transmission", [])

        carrier = ViraxCarrier()
        extracted_frags = []
        for i, text in enumerate(transmission):
            extracted = carrier.extract(text)
            if extracted:
                extracted_frags.append(extracted)
                print(f"  Fragment {i}: ✓")
            else:
                print(f"  Fragment {i}: ✗")

        print("[+] Reconstructing message...")
        try:
            reconstructed = ViraxEntropy.reconstruct(extracted_frags)
            print("[+] Decrypting...")
            decrypted, timestamp = ViraxProtocol.decrypt_message(
                receiver.enc_priv,
                sender_info["sign_pub"],
                reconstructed
            )
            print("\n[✓] MESSAGE DECRYPTED!")
            print(f"From: {sender_info['identity']}")
            print(f"Timestamp: {time.ctime(timestamp)}")
            print(f"Content: {decrypted}")
        except ValueError as e:
            print(f"\n[-] ERROR: {e}")
        return

    # Help
    print("VIRAXTUNNEL V3 - Furtive communication system")
    print("Use --help to see available options")
    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[-] ERROR: {e}")
        if os.environ.get("DEBUG"):
            import traceback
            traceback.print_exc()
        sys.exit(1)
