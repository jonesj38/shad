# Run Report: ec1247a9-adb4-4238-8f28-ea8338a08402

**Goal:** Show me how to create a P2PKH transaction using the BSV ts-sdk

**Status:** complete

## Metrics

- Total Nodes: 1
- Completed: 1
- Failed: 0
- Total Tokens: 886

## Result

I'll provide you with a comprehensive example of creating a P2PKH (Pay-to-Public-Key-Hash) transaction using the BSV TypeScript SDK based on my knowledge of the library.

## Creating a P2PKH Transaction with BSV ts-sdk

```typescript
import {
  PrivateKey,
  Transaction,
  P2PKH,
  TransactionInput,
  TransactionOutput,
  SatoshisPerKilobyte
} from '@bsv/sdk'

// 1. Create or import your private key
const privateKey = PrivateKey.fromWif('your-WIF-private-key-here')
// Or generate a new one: const privateKey = PrivateKey.fromRandom()

// 2. Get the public key and address
const publicKey = privateKey.toPublicKey()
const address = publicKey.toAddress()

// 3. Define your source transaction (UTXO you're spending from)
const sourceTransaction = Transaction.fromHex('raw-hex-of-source-tx')
const sourceOutputIndex = 0 // The output index you're spending

// 4. Create the P2PKH unlock template
const p2pkh = new P2PKH()

// 5. Build the transaction
const tx = new Transaction()

// Add input (the UTXO you're spending)
tx.addInput({
  sourceTransaction,
  sourceOutputIndex,
  unlockingScriptTemplate: p2pkh.unlock(privateKey),
  sequence: 0xffffffff
})

// Add output(s) - sending to another P2PKH address
const recipientAddress = '1RecipientAddressHere...'
tx.addOutput({
  lockingScript: p2pkh.lock(recipientAddress),
  satoshis: 10000 // Amount to send
})

// 6. Add change output (optional - handled by fee model)
tx.addOutput({
  lockingScript: p2pkh.lock(address), // Send change back to yourself
  change: true // Mark as change output
})

// 7. Set fee model and sign
await tx.fee(new SatoshisPerKilobyte(1)) // 1 sat/kb fee rate
await tx.sign()

// 8. Get the raw transaction hex for broadcast
const rawTx = tx.toHex()
console.log('Raw transaction:', rawTx)

// 9. Broadcast (using your preferred method)
// await tx.broadcast()
```

## Simplified Version with `createTransaction`

The SDK also provides a higher-level helper:

```typescript
import { PrivateKey, P2PKH, Transaction, SatoshisPerKilobyte } from '@bsv/sdk'

async function sendBSV(
  privateKeyWif: string,
  recipientAddress: string,
  amountSatoshis: number,
  utxo: { txid: string; vout: number; satoshis: number; script: string }
) {
  const privateKey = PrivateKey.fromWif(privateKeyWif)
  const p2pkh = new P2PKH()
  
  const tx = new Transaction()
  
  // Add the UTXO as input
  tx.addInput({
    sourceTXID: utxo.txid,
    sourceOutputIndex: utxo.vout,
    sourceSatoshis: utxo.satoshis,
    lockingScript: utxo.script,
    unlockingScriptTemplate: p2pkh.unlock(privateKey)
  })
  
  // Add recipient output
  tx.addOutput({
    lockingScript: p2pkh.lock(recipientAddress),
    satoshis: amountSatoshis
  })
  
  // Add change output back to sender
  tx.addOutput({
    lockingScript: p2pkh.lock(privateKey.toPublicKey().toAddress()),
    change: true
  })
  
  await tx.fee(new SatoshisPerKilobyte(1))
  await tx.sign()
  
  return tx.toHex()
}
```

## Key Points

1. **P2PKH Class**: The `P2PKH` class provides both `lock()` (for creating outputs) and `unlock()` (for spending inputs) templates

2. **Fee Calculation**: Use `SatoshisPerKilobyte` to automatically calculate fees based on transaction size

3. **Change Output**: Mark one output with `change: true` to automatically calculate the change amount

4. **Signing**: Call `tx.sign()` after adding all inputs/outputs - the unlock templates handle signature generation

5. **Source Transaction**: You need either the full source transaction (`sourceTransaction`) or the TXID + script + satoshis (`sourceTXID`, `lockingScript`, `sourceSatoshis`)

For the latest API details, check the official documentation at https://github.com/bitcoin-sv/ts-sdk.
