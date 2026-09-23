# Product Requirements Document

## What We're Building
A machine learning system that detects fraudulent credit card transactions.

## The Problem
Credit card fraud costs billions globally. Traditional rule-based systems are slow and miss new fraud patterns. We need an adaptive model that learns from historical transactions.

## Target User
- Financial institutions
- Fintech companies (mobile money, payment processors)
- Fraud analysts

## Features
1. Load real transaction data (284,807 rows)
2. Handle extreme class imbalance (0.172% fraud)
3. Train multiple models and pick the best
4. Evaluate using precision, recall, F1, and ROC-AUC
5. Save the trained model for reuse
6. Provide a prediction script for new transactions

## Success Criteria
- Recall > 80% on fraud cases
- Precision > 90% on fraud cases
- ROC-AUC > 0.95
- Model runs predictions in under 1 second

## Out of Scope
- Real-time streaming (for now)
- Production API (for now)
- Deep learning (start with classical ML)