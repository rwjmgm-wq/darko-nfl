# Using cfbfastR as Alternative to CFBD Python API

Since the CFBD Python API is hitting persistent 401 AUTH errors, we're trying cfbfastR (the R equivalent of nflfastR) as an alternative data source.

## Why cfbfastR?

- Uses same CFBD API but through R implementation
- May have different rate limiting/auth handling
- Actively maintained by SportsDataverse
- More robust caching mechanisms
- Similar interface to nflfastR (which worked perfectly for NFL data)

## Prerequisites

You need R installed on your system.

### Install R (if not already installed)

**Windows:**
1. Download R from: https://cran.r-project.org/bin/windows/base/
2. Run the installer
3. Default options are fine

**Or use Chocolatey:**
```bash
choco install r.project
```

## Step 1: Install R Dependencies

Run this to install cfbfastR and required packages:

```bash
cd "c:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\college_to_nfl_projection"
Rscript src/install_r_dependencies.R
```

This will:
- Install dplyr, tidyr, readr from CRAN
- Install cfbfastR from GitHub
- Test the connection with a sample query

## Step 2: Collect College Career Data

Once dependencies are installed, run:

```bash
cd "c:\Users\rwjmg\OneDrive\Pictures\Writing\DARKO_NFL\college_to_nfl_projection"
Rscript src/collect_college_careers_cfbfastr.R
```

This will:
- Load the 124 QB list
- Fetch full college career play-by-play for each QB
- Save individual career files (one per QB)
- Create a career_summary.csv
- Save everything to: `data/processed/full_college_careers/`

## Expected Output

```
Loading QB list...
Loaded 124 QBs

Collecting career data...
================================================================================
Processing: JaMarcus Russell (LSU, 2007)
  Fetching 2003...
  Fetching 2004...
  Fetching 2005...
    Found 234 plays
  Fetching 2006...
    Found 401 plays
  [1/124] JaMarcus Russell: SUCCESS (635 plays)

Processing: Brady Quinn (Notre Dame, 2007)
  ...
```

## If It Works

If cfbfastR successfully collects the data, you'll have:

- `data/processed/full_college_careers/*.csv` - Individual QB career files
- `data/processed/full_college_careers/career_summary.csv` - Summary of all QBs

Then you can proceed with the analysis:

```bash
# Step 3: Create aggregations
python src/create_multi_year_aggregations.py

# Step 4: Test models
python src/test_comprehensive_models.py

# Step 5: Generate report
python src/generate_comprehensive_report.py
```

## Advantages of cfbfastR

1. **Built-in caching** - Automatically caches data locally
2. **Better error handling** - More robust retry mechanisms
3. **Cleaner data** - Pre-processed and validated
4. **Active development** - Regular updates and bug fixes
5. **Consistent with nflfastR** - Same team, same patterns

## If cfbfastR Also Fails

If cfbfastR hits the same auth issues, the problem is with the CFBD API itself (not the wrapper). In that case:

**Option A:** Wait for CFBD API to resolve issues
**Option B:** Use existing data (62 QBs instead of 124)
**Option C:** Manual data collection from Sports Reference

## Troubleshooting

**"Rscript not found"**
- R is not in your PATH
- Try: `"C:\Program Files\R\R-4.X.X\bin\Rscript.exe"` (replace X.X with your version)
- Or reinstall R and select "Add to PATH" option

**"cfbfastR install fails"**
- Make sure devtools can build packages
- Windows: Install Rtools from https://cran.r-project.org/bin/windows/Rtools/
- Mac: Install Xcode Command Line Tools

**"Still getting 401 errors"**
- CFBD API key may need to be set
- Try setting environment variable: `CFBD_API_KEY=your_key`
- Contact CFBD support about persistent auth issues

## Comparison: Python cfbd vs R cfbfastR

| Feature | Python cfbd | R cfbfastR |
|---------|-------------|-------------|
| Install | pip | GitHub |
| Auth | Manual API key | Auto-handled |
| Caching | Manual | Automatic |
| Rate limiting | Manual retry | Built-in |
| Data cleaning | Manual | Automatic |
| Status | Failing (401) | **Testing now** |

Let's see if cfbfastR can get through where the Python library couldn't!
