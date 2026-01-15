# Run Report: b6fb7428-4ea7-4dac-a1c5-609c8ab5fb89

**Goal:** How do we use the BSV ts-sdk library to build a BRC-100 compliant transaction?

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 0
- Failed: 0
- Total Tokens: 0

## Result

I don't have web access enabled to fetch the latest documentation. However, I can provide you with guidance based on my knowledge of the BSV ts-sdk and BRC-100 standard.

## Building a BRC-100 Compliant Transaction with BSV ts-sdk

**BRC-100** defines the standard transaction format for the BSV ecosystem, focusing on proper input/output construction, signature handling, and transaction serialization.

### Basic Transaction Building Pattern

```typescript
import { 
  Transaction, 
  PrivateKey, 
  P2PKH,
  SatoshisPerKilobyte
} from '@bsv/sdk'

// 1. Create a new transaction
const tx = new Transaction()

// 2. Add inputs (UTXOs you're spending)
tx.addInput({
  sourceTXID: 'previous_txid_hex',
  sourceOutputIndex: 0,
  sourceSatoshis: 10000,
  unlockingScriptTemplate: new P2PKH().unlock(privateKey)
})

// 3. Add outputs
tx.addOutput({
  lockingScript: new P2PKH().lock(recipientAddress),
  satoshis: 5000
})

// 4. Set fee model (BRC-100 compliant fee calculation)
tx.fee(new SatoshisPerKilobyte(1)) // 1 sat/kb is standard

// 5. Sign the transaction
await tx.sign()

// 6. Serialize for broadcast
const rawTx = tx.toHex()
```

### Key BRC-100 Compliance Points

1. **Proper TXID Format**: Use 64-character lowercase hex strings for transaction IDs
2. **Correct Satoshi Values**: Use integer satoshis, not decimal BSV
3. **Standard Fee Calculation**: Use `SatoshisPerKilobyte` fee model
4. **Canonical Signatures**: The SDK handles low-S signatures automatically
5. **Input Validation**: Always specify `sourceSatoshis` for proper fee calculation

### With Source Transaction Lookup

```typescript
import { Transaction, WhatsOnChain } from '@bsv/sdk'

// Use a chain tracker for automatic UTXO resolution
const chainTracker = new WhatsOnChain('main')

const tx = new Transaction()
await tx.addInput({
  sourceTXID: 'txid_here',
  sourceOutputIndex: 0,
  unlockingScriptTemplate: new P2PKH().unlock(privateKey)
}, chainTracker) // Automatically fetches source tx details

// ... add outputs and sign
```

### Resources

For the latest API details, check:
- **GitHub**: `github.com/bitcoin-sv/ts-sdk`
- **NPM**: `@bsv/sdk`
- **BRC Specs**: `github.com/bitcoin-sv/BRCs` (contains all BRC standards)

Would you like me to elaborate on any specific aspect, such as OP_RETURN data outputs, multi-signature transactions, or custom locking scripts?
