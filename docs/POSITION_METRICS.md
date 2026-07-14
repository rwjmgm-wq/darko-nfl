# Position Metrics Availability Analysis

**Analysis Date**: 2025-11-05
**Data Source**: nflfastR play-by-play (2023-2024 seasons)
**Total Plays Analyzed**: 99,157
**Available Columns**: 397

---

## Executive Summary

nflfastR provides **excellent direct EPA attribution** for offensive skill positions (QB, RB, WR/TE), but **limited individual player data** for offensive line and defensive positions. Multi-position LEAF is feasible for skill positions first, with OL and defense requiring inference methods.

### Data Quality Tiers

| Tier | Positions | Sample Size | EPA Attribution | Implementation |
|------|-----------|-------------|-----------------|----------------|
| **EXCELLENT** | QB | 41K plays, 161 players | Direct | Phase 2 (done) |
| **GOOD** | WR/TE, RB | 36K/31K plays, 600+ players | Direct | Phase 3-4 |
| **FAIR** | OL | Team-level only | Inferred | Phase 4b |
| **FAIR** | Defense | Team-level + tackles | Team ratings | Phase 5 |

---

## 1. Quarterback (QB)

### Data Quality: **EXCELLENT**

### Available Metrics

**Sample Size (2023-2024)**:
- Total pass attempts: **40,879**
- Unique QBs: **161**
- Average plays per QB: **254**
- Example: Patrick Mahomes (1,495 passes)

**Direct EPA Attribution**:
- `qb_epa`: ✅ 100% coverage
- `cpoe`: ✅ 88.4% coverage (completion probability over expected)
- `success`: ✅ 100% coverage
- `air_yards`: ✅ 92.5% coverage
- `yards_after_catch`: ✅ 60.1% coverage

**Traditional Stats**:
- Completions, attempts, TDs, INTs: ✅ Full coverage
- Sacks, QB hits, scrambles: ✅ Full coverage

**Derivable Metrics**:
- Adjusted EPA (opponent, context, Kalman): ✅ Already implemented
- Success rate, CPOE: ✅ Already implemented
- Time-weighted metrics: ✅ Already implemented

### Implementation Status
✅ **Complete** - QB LEAF fully functional with empirically validated weights

### Recommendations
- **Ready for teammate adjustments** - can now adjust QB EPA for receiver/OL quality

---

## 2. Wide Receivers & Tight Ends (WR/TE)

### Data Quality: **GOOD**

### Available Metrics

**Sample Size (2023-2024)**:
- Total targets: **36,298**
- Unique receivers: **634**
- Average targets per receiver: **57.3**
- Example: CeeDee Lamb (353 targets)

**Direct EPA Attribution**:
- `epa`: ✅ 100% coverage (EPA on plays where targeted)
- `air_yards`: ✅ 99.5% coverage
- `yards_after_catch`: ✅ 67.7% coverage
- Target data: ✅ 100% (complete_pass, incomplete_pass flags)

**Derivable Metrics**:
- **Catch Rate**: targets vs completions ✅
- **RACR** (Receiver Air Conversion Ratio): YAC / air yards ✅
- **Target Share**: % of team targets ✅
- **EPA per target**: Direct from play-by-play ✅
- **Yards per route run**: ⚠️ Requires route data (PFF or NGS)
- **Separation**: ⚠️ Requires Next Gen Stats

**Position Split**:
- WR vs TE distinction requires matching with roster data ✅ (available)
- Slot vs outside: ⚠️ Requires formation data or PFF

### Sample Sizes by Threshold

| Threshold | Players | Notes |
|-----------|---------|-------|
| 30+ targets | ~300-350 | Adequate for LEAF |
| 50+ targets | ~200-250 | Good confidence |
| 100+ targets | ~80-100 | High confidence |

### Implementation Priority: **HIGH** (Phase 3)

### Recommendations
1. Start with **combined WR/TE** analysis (split later if needed)
2. Minimum threshold: **30 targets** for LEAF inclusion
3. Primary metrics: **EPA per target**, **catch rate**, **RACR**
4. Test: Does adjusting QB for WR quality improve prediction?

---

## 3. Running Backs (RB)

### Data Quality: **GOOD**

### Available Metrics

**Sample Size (2023-2024)**:
- Total rush attempts: **30,813**
- Unique rushers: **463**
- Average rushes per player: **66.6**
- RB targets (receiving): **Included in 36K receiver sample**

**Direct EPA Attribution**:
- **Rushing EPA**: ✅ 100% coverage
- **Receiving EPA**: ✅ 100% coverage (when targeted)
- `yards_gained`: ✅ 100% coverage
- `success`: ✅ Available for rushes

**Derivable Metrics**:
- **Rush EPA per attempt**: Direct ✅
- **Receiving EPA per target**: Direct ✅
- **Composite RB value**: Rush + receiving EPA ✅
- **Yards before contact**: ⚠️ Requires Next Gen Stats or PFF
- **Broken tackles**: ⚠️ Requires PFF
- **Pass blocking**: ⚠️ Must infer from QB pressure when RB in protection

### RB Pass Blocking (Inference Method)
- Identify plays where RB is in pass protection (formation-based heuristic)
- Calculate QB pressure/sack rate on those plays
- Attribute protection quality to RB (team-level initially)

### Implementation Priority: **MEDIUM-HIGH** (Phase 4a)

### Recommendations
1. Start with **rush + receiving EPA** combined
2. Minimum threshold: **40 total touches** (rushes + targets)
3. Pass blocking as **optional enhancement** (Phase 4b)
4. Test: Does adjusting QB for RB quality matter? (May be minimal)

---

## 4. Offensive Line (OL)

### Data Quality: **FAIR** (Inference Required)

### Challenge
**No direct player-level EPA attribution** in nflfastR. Must reverse-engineer from QB/RB performance.

### Inference Methodology

#### Pass Protection Quality
**Available indicators**:
- `sack`: ✅ Team-level sack rate
- `qb_hit`: ✅ QB hits recorded
- `qb_hurry`: ⚠️ Limited coverage
- Time to throw: ⚠️ Requires Next Gen Stats

**Inference approach**:
1. Calculate team-level pass protection quality
2. Adjust for QB tendencies (holding ball, scrambling)
3. Use opponent pass rush quality (from defense ratings)
4. Residual = OL pass protection contribution

#### Run Blocking Quality
**Available indicators**:
- `yards_gained`: ✅ Full coverage
- Stuff rate (gain <= 0 yards): ✅ Derivable
- Success rate on runs: ✅ Available

**Inference approach**:
1. Calculate team-level run blocking quality
2. Adjust for RB quality (chicken-egg problem - iterative solution)
3. Adjust for opponent run defense
4. Residual = OL run blocking contribution

### Implementation Priority: **MEDIUM** (Phase 4b)

### Recommendations
1. **Team-level OL ratings initially** (not individual linemen)
2. Use **iterative convergence** (like defense ratings)
3. Separate **pass protection** and **run blocking** ratings
4. Test: Does QB-OL-RB triangle improve predictions?
5. **Future enhancement**: PFF has individual OL grades and pressure data

---

## 5. Defensive Positions

### Data Quality: **FAIR** (Limited Individual Attribution)

### Available Data

**Tackle Data** (Individual Player Level):
- `solo_tackle_1_player_id/name`: ✅ Primary tackler
- `assist_tackle_1-4_player_id/name`: ✅ Assisted tackles
- `tackle_for_loss_player_id/name`: ✅ TFLs tracked

**Pass Defense Data**:
- `pass_defense_1_player_id/name`: ✅ PBUs/deflections
- `interception_player_id`: ✅ (via turnover columns)
- Coverage metrics: ❌ Not available (need PFF or player tracking)

**Formation Data**:
- `defense_personnel`: ✅ Personnel grouping
- `defense_coverage_type`: ⚠️ Limited coverage (~20-30%)
- `defense_man_zone_type`: ⚠️ Limited coverage

### Current Defense Rating System

**Already Implemented** (from opponent_adjustments.py):
- Team-level defense EPA ratings ✅
- Iterative convergence with offense ✅
- Adjusts QB/RB/WR for opponent defense quality ✅

### Individual Defensive Player Approach

**Defensive Line (DL/EDGE)**:
- Infer from: QB sacks, QB hits, QB hurries
- Assign credit using tackle data
- Team pass rush win rate (from QB pressure)

**Linebackers (LB)**:
- Run defense: Tackle data on run plays
- Coverage: Pass defense events, EPA when targeted
- Blitz: Pressure when blitzing (requires alignment data)

**Defensive Backs (DB)**:
- Coverage EPA: Targets allowed, completions, yards
- Requires matching receiver to nearest defender (tracking data)
- Interceptions and PBUs: ✅ Available
- Completion % allowed: Derivable with roster matching

### Implementation Priority: **MEDIUM-LOW** (Phase 5)

### Recommendations
1. **Phase 5a**: Extend team defense ratings to position groups (DL, LB, DB)
2. **Phase 5b**: Individual defenders (limited to tackle/turnover data)
3. **Future**: Integrate player tracking for coverage metrics
4. **Alternative**: Use existing team defense ratings for now

---

## 6. Column Inventory

### Key nflfastR Columns for LEAF

#### Essential
- `play_id`, `game_id`, `season`, `week`
- `posteam`, `defteam`
- `play_type`, `pass_attempt`, `rush_attempt`
- `epa`, `success`

#### Quarterback
- `passer_player_id`, `passer_player_name`
- `qb_epa`, `cpoe`
- `complete_pass`, `incomplete_pass`, `touchdown`
- `interception`, `sack`, `qb_hit`

#### Receivers
- `receiver_player_id`, `receiver_player_name`
- `air_yards`, `yards_after_catch`
- `complete_pass`, `touchdown`

#### Running Backs
- `rusher_player_id`, `rusher_player_name`
- `yards_gained`, `touchdown`

#### Context
- `down`, `ydstogo`, `yardline_100`
- `score_differential`, `qtr`, `half_seconds_remaining`
- `wp` (win probability), `vegas_wp`

#### Opponent Adjustments
- `posteam`, `defteam` (for defense ratings)

### Full Column List
See [nflfastr_columns.csv](nflfastr_columns.csv) for complete inventory (397 columns)

---

## 7. Sample Size Requirements

### Recommended Minimum Thresholds for LEAF

Based on sample sizes and confidence intervals:

| Position | Minimum Plays | Reasoning |
|----------|--------------|-----------|
| QB | 100 attempts | ~6-7 games, adequate for trend |
| WR/TE | 30 targets | ~2-3 games as WR1/2, basic signal |
| RB | 40 touches | Rush + receiving, ~3-4 games |
| OL | Team-level | 17 games (full season) |
| Defense | Team-level | 17 games (full season) |

### Confidence Tiers

| Plays | Confidence | Usage |
|-------|-----------|-------|
| < 50 | Very Low | Exclude from LEAF |
| 50-100 | Low | Include with heavy regression |
| 100-200 | Medium | Include with moderate regression |
| 200+ | High | Full weight |

This matches our Bayesian approach - small samples regress toward league mean.

---

## 8. Implementation Roadmap

### Phase 1: Data Foundation ✅ COMPLETE
- [x] Explore nflfastR data availability
- [x] Document metrics by position
- [x] Identify sample sizes
- [x] Assess data quality

### Phase 2: Infrastructure (NEXT)
- [ ] Build position interaction matrix
- [ ] Create teammate quality calculator
- [ ] Implement iterative adjustment solver

### Phase 3: QB + Receivers (POC)
- [ ] Aggregate receiver EPA per target
- [ ] QB ↔ WR teammate adjustments
- [ ] Validate on 2020-2022 → 2023
- [ ] **Decision point**: Proceed only if improvement > 1%

### Phase 4a: Add RB
- [ ] Aggregate RB rush + receiving EPA
- [ ] QB ↔ RB, OL ↔ RB adjustments
- [ ] Validate improvement

### Phase 4b: Add OL (Team-Level)
- [ ] Infer OL pass protection from QB performance
- [ ] Infer OL run blocking from RB performance
- [ ] Iterative QB-OL-RB solution

### Phase 5: Defense
- [ ] Extend team defense ratings
- [ ] Position group defense ratings (DL, LB, DB)
- [ ] Individual defenders (limited metrics)

### Phase 6: Universal LEAF
- [ ] Unified rating system across all positions
- [ ] Common EPA scale
- [ ] Position-specific plays-per-game conversions
- [ ] Universal WAR calculation

---

## 9. Data Limitations & Mitigations

### Limitations

1. **No individual OL data**
   - Mitigation: Team-level inference, future PFF integration

2. **Limited defensive player tracking**
   - Mitigation: Use team defense ratings, focus on tackles/turnovers

3. **No route running data**
   - Mitigation: Use target-based metrics initially

4. **Coverage assignments not tracked**
   - Mitigation: Nearest defender heuristic, future player tracking

5. **Context/formation data incomplete**
   - Mitigation: Use available context adjustments (weather, script, down/distance)

### Future Enhancements

**Next Gen Stats Integration**:
- Player tracking data (speed, separation, time to throw)
- Route depth and route type
- Pressure data with pass rush win rate

**PFF Integration**:
- Individual OL grades
- Coverage grades
- Big-time throws / turnover-worthy plays

---

## 10. Key Takeaways

1. **Skill positions (QB, WR/TE, RB) are ready** for multi-position LEAF
   - Excellent data quality
   - Direct EPA attribution
   - Sufficient sample sizes

2. **OL requires inference** but is feasible
   - Team-level initially
   - Iterative solution for QB-OL-RB interactions

3. **Defense is limited** to team-level + basic individual stats
   - Current defense rating system is solid foundation
   - Individual defenders: future enhancement

4. **Phase 3 is low-risk**: QB + WR has excellent data
   - Perfect proof-of-concept for teammate adjustments
   - If this doesn't work, nothing will

5. **Sample size management is critical**
   - Use Bayesian regression for small samples
   - Set clear minimum thresholds
   - Weight by confidence (like current LEAF uncertainty adjustment)

---

## Next Steps

1. ✅ **Phase 1 Complete**: Data exploration done
2. ⏭️ **Phase 2**: Build interaction matrix infrastructure
3. ⏭️ **Phase 3**: Implement QB ↔ WR teammate adjustments
4. 📊 **Validation**: Test every adjustment before including

**Ready to proceed with Phase 2: Infrastructure development.**
