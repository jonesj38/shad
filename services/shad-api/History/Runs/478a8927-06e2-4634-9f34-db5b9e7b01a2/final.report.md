# Run Report: 478a8927-06e2-4634-9f34-db5b9e7b01a2

**Goal:** Show me how to create a P2PKH transaction using the BSV ts-sdk

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 732

## Result

# Creating a P2PKH Transaction with BSV ts-sdk

Based on the BSV blockchain TypeScript SDK structure, here's how to create a Pay-to-Public-Key-Hash (P2PKH) transaction:

## Basic P2PKH Transaction

```typescript
import { PrivateKey, P2PKH, Transaction } from '@bsv/sdk'

// 1. Create or import your private key
const privateKey = PrivateKey.fromWif('your-WIF-private-key')
// Or generate a new one:
// const privateKey = PrivateKey.fromRandom()

// 2. Get the public key and address
const publicKey = privateKey.toPublicKey()

// 3. Create a P2PKH template
const p2pkh = new P2PKH()

// 4. Build the transaction
const tx = new Transaction()

// 5. Add input (spending from a previous UTXO)
tx.addInput({
  sourceTXID: 'previous-transaction-id-hex',
  sourceOutputIndex: 0,
  unlockingScriptTemplate: p2pkh.unlock(privateKey)
})

// 6. Add output (sending to a recipient)
tx.addOutput({
  lockingScript: p2pkh.lock(recipientPublicKeyHash),
  satoshis: 1000
})

// 7. Sign and finalize the transaction
await tx.sign()

// 8. Get the raw transaction hex for broadcasting
const rawTx = tx.toHex()
```

## Complete Example with UTXO

```typescript
import { PrivateKey, P2PKH, Transaction, Hash } from '@bsv/sdk'

async function createP2PKHTransaction() {
  // Sender's private key
  const senderPrivateKey = PrivateKey.fromWif('L1...')
  const senderPublicKey = senderPrivateKey.toPublicKey()
  
  // Recipient's public key hash (address)
  const recipientPubKeyHash = Hash.hash160(recipientPublicKey.toBytes())
  
  // P2PKH template
  const p2pkh = new P2PKH()
  
  // Create transaction
  const tx = new Transaction()
  
  // Add input from UTXO you want to spend
  tx.addInput({
    sourceTXID: 'abc123...', // Previous tx ID
    sourceOutputIndex: 0,    // Output index in previous tx
    unlockingScriptTemplate: p2pkh.unlock(senderPrivateKey),
    sequence: 0xffffffff
  })
  
  // Add output to recipient
  tx.addOutput({
    lockingScript: p2pkh.lock(recipientPubKeyHash),
    satoshis: 5000
  })
  
  // Add change output back to sender (optional)
  tx.addOutput({
    lockingScript: p2pkh.lock(Hash.hash160(senderPublicKey.toBytes())),
    satoshis: 4500 // Original amount minus payment minus fee
  })
  
  // Sign the transaction
  await tx.sign()
  
  return tx
}
```

## Key Components

| Component | File Location | Purpose |
|-----------|---------------|---------|
| `P2PKH` | `src/script/templates/P2PKH.ts` | P2PKH locking/unlocking template |
| `PrivateKey` | `src/primitives/PrivateKey.ts` | Key management |
| `Transaction` | `src/transaction/` | Transaction building |

## Broadcasting

After creating the transaction, broadcast it using the ARC broadcaster:

```typescript
import { ARC } from '@bsv/sdk'

const broadcaster = new ARC('https://arc.taal.com')
const result = await broadcaster.broadcast(tx)
```

**Source**: [[github.com_bsv-blockchain_ts-sdk]] - The SDK provides P2PKH templates in `src/script/templates/P2PKH.ts` along with transaction building utilities in the `transaction` module and cryptographic primitives in `src/primitives/`.
