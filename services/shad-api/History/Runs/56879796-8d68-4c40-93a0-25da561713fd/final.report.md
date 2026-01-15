# Run Report: 56879796-8d68-4c40-93a0-25da561713fd

**Goal:** What classes and methods does the BSV ts-sdk provide for transaction signing?

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 598

## Result

Based on my knowledge of the BSV TypeScript SDK (ts-sdk), here are the key classes and methods for transaction signing:

## Core Signing Classes

### **Transaction**
The main class for building and signing transactions:
- `sign()` - Signs all inputs using provided private keys
- `addInput()` / `addOutput()` - Build the transaction structure

### **PrivateKey**
Used for cryptographic signing operations:
- `sign(message)` - Signs a message/hash with the private key
- `toPublicKey()` - Derives the corresponding public key
- `fromWif(wifString)` - Creates from Wallet Import Format

### **PublicKey**
For signature verification:
- `verify(message, signature)` - Verifies a signature against a message

### **Signature**
Represents ECDSA signatures:
- `toDER()` - Serializes to DER format
- `fromDER()` - Parses from DER format

### **TransactionSignature**
Specialized for transaction input signing:
- Combines the signature with sighash flags (e.g., `SIGHASH_ALL`, `SIGHASH_FORKID`)

## Unlocking Script Templates

The SDK uses a template pattern for different script types:

### **P2PKH (Pay-to-Public-Key-Hash)**
- `UnlockingScript.fromP2PKH(privateKey)` - Creates standard unlocking script

### **Custom Script Signing**
- `Script.fromASM()` / `Script.fromHex()` - For custom locking/unlocking scripts

## Sighash Flags
Constants for controlling what parts of the transaction are signed:
- `Signature.SIGHASH_ALL` - Sign all inputs and outputs
- `Signature.SIGHASH_NONE` - Sign inputs only
- `Signature.SIGHASH_SINGLE` - Sign corresponding output only
- `Signature.SIGHASH_ANYONECANPAY` - Only sign the current input
- `Signature.SIGHASH_FORKID` - BSV-specific fork ID (required for BSV)

## Typical Signing Flow

```typescript
import { PrivateKey, Transaction, P2PKH } from '@bsv/sdk'

const privateKey = PrivateKey.fromWif('...')
const tx = new Transaction()
tx.addInput({
  sourceTXID: '...',
  sourceOutputIndex: 0,
  unlockingScriptTemplate: new P2PKH().unlock(privateKey)
})
tx.addOutput({
  lockingScript: new P2PKH().lock(address),
  satoshis: 1000
})
await tx.sign()
```

For the most current and detailed API documentation, I recommend checking the official BSV ts-sdk repository at https://github.com/bitcoin-sv/ts-sdk or the official documentation site, as the SDK may have evolved since my knowledge cutoff.
