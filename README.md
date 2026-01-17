![Pylint Status](https://github.com/damienos61/viraxtunnel/actions/workflows/pylint.yml/badge.svg)

# 🛡️ VIRAXTUNNEL V3 — ULTIMATE BATTLE-READY

ViraxTunnel is a stealth communication system designed for environments where network surveillance is pervasive. It combines the strength of modern asymmetric cryptography with the subtlety of Unicode steganography.

"The channel is dead. Only information survives in the chaos."

---

## 🚀 Key Features

* **Spectral Identity**: Generation of hybrid asymmetric keys (Ed25519 for signing, RSA-3072 for encryption).
* **Military-Grade Encryption**: AES-256-GCM for data, wrapped with RSA-OAEP for secure key transmission.
* **Invisible Steganography**: Embeds data into ordinary cover text using non-printable Unicode characters.
* **Entropy Resilience**: Message fragmentation with configurable redundancy and majority-vote reconstruction to protect against fragment loss.
* **Operational Security**: Atomic file writes and Time-To-Live (TTL) management for message expiration.

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/viraxtunnel.git
cd viraxtunnel
```

2. Install dependencies:

```bash
pip install cryptography
```

---

## 📖 Usage Guide (Alice & Bob Scenario)

### 1. Identity Initialization

Alice and Bob generate their respective identities:

```bash
# Alice
python viraxtunnel.py --new-identity alice --password "secret_alice"

# Bob
python viraxtunnel.py --new-identity bob --password "secret_bob"
```

### 2. Public Key Exchange

Each user must export their public bundle and send it to their correspondent:

```bash
python viraxtunnel.py --identity alice --get-public > alice_pub.json
```

### 3. Sending a Message (Alice)

Alice encrypts a message for Bob using his public bundle:

```bash
python viraxtunnel.py --identity alice --password "secret_alice" --receiver-pub bob_pub.json --message "Meet at the safehouse at 22:00"
```

This generates `transmission.json`, ready to be sent through any channel (email, Discord, chat).

### 4. Receiving and Decrypting a Message (Bob)

Bob retrieves the transmission file and decrypts the secret using his private key:

```bash
python viraxtunnel.py --identity bob --password "secret_bob" --decrypt transmission.json --sender-pub alice_pub.json
```

---

## 🔬 Technical Architecture

* **Data Flow:**

  * **Compression**: Zlib (reduces footprint)
  * **Encryption**: AES-256-GCM + RSA-OAEP
  * **Signature**: Ed25519 (ensures authenticity)
  * **Fragmentation**: Splits messages into N chunks with SHA-1 checksums
  * **Injection**: Embeds fragments into Unicode characters from `\u200b` to `\ufeff`

* **Carrier Design:**

  * Multi-insertion into cover text to reduce fragment loss.
  * CRC32 checksum verification for each fragment.

* **Protocol Flow:**

  1. Sender compresses and encrypts the message.
  2. AES key is wrapped with receiver's RSA public key.
  3. Packet is signed with sender's Ed25519 private key.
  4. Packet is fragmented and embedded into cover text using ViraxCarrier.
  5. Receiver extracts fragments, reconstructs the packet, verifies signature, unwraps AES key, decrypts, and validates TTL.

---

## ⚠️ Legal Disclaimer

This software is provided for educational and research purposes only. The author is not responsible for any malicious or illegal use of this tool. Absolute security does not exist; endpoint protection (your PC) remains your responsibility.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open a discussion first to propose improvements.

**License:** MIT
