```mermaid
flowchart LR
    subgraph Setup
        A[Raw Data] --> B[Data Preprocessing & Encoding];
    end
    
    subgraph Batch Processing & Causal Analysis
        B --> C[Loop: Batch Start];
        
        C --> D{LLM Inference Check/Run<br>Test or Batch Preds};
        D --> E[Merge Predictions<br>On JHNT_MBN];
        E --> F[Causal Estimation<br>ATE 추정 and Refutation];
        F --> G[Save Batch Results<br>batch_results_i.csv];
        
        G --> C; 
    end
    
    subgraph Final Reporting
        G --> H[End Batch Loop];
        H --> I[Final Consolidation<br>all_validation_results.csv];
        I --> J[Final Interpretation<br>전체 통계 및 안정성 해석];
        J --> K[End Pipeline];
    end