#!/usr/bin/env python3
"""
Summary of the new seizure validation feature for patient selection.

This module implements seizure validation criteria when selecting top patients
based on seizure count. Only "valid" seizures are counted.

SEIZURE VALIDATION CRITERIA:
1. Temporal Criterion: Seizure must occur at least 20 minutes from recording start
2. Post-ictal Criterion: Seizure must not be within 30-minute post-ictal refractory 
   period of another valid seizure

IMPLEMENTATION DETAILS:
- Located in DataProcessingPipeline._validate_seizures()
- Used by DataProcessingPipeline._count_seizures_per_patient()
- Affects patient ranking when using --top-n-patients argument

EXAMPLE USAGE:
python3 data_processing_pipeline.py --top-n-patients 10

This will:
1. Load all seizure annotations for all patients
2. Apply validation criteria to each seizure
3. Count only valid seizures per patient  
4. Select top 10 patients with most valid seizures
5. Process only recordings from selected patients

VALIDATION EXAMPLES:
✓ Valid: Seizure at 30min (>20min from start, first seizure)
✓ Valid: Seizure at 90min (>20min from start, >30min from previous)
✗ Invalid: Seizure at 10min (too early - <20min from start)
✗ Invalid: Seizure at 45min if previous at 30min (within 30min post-ictal)

BENEFITS:
- Reduces noise from early recording artifacts
- Respects neurological refractory periods
- Improves patient selection quality for seizure prediction models
- Provides more reliable seizure counts for research
"""

if __name__ == "__main__":
    print(__doc__)