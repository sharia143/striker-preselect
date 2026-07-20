-- Aggregate player-match data to player-season level for forwards/strikers
-- Metrics are per-90-minutes rates

SELECT
    player_id,
    player,
    team,
    primary_position,
    COUNT(DISTINCT match_id) AS matches,
    SUM(minutes) AS total_minutes,
    -- Per 90 metrics
    SUM(shots) / (SUM(minutes) / 90.0) AS shots_p90,
    SUM(npxg) / (SUM(minutes) / 90.0) AS npxg_p90,
    SUM(touches_in_box) / (SUM(minutes) / 90.0) AS touches_in_box_p90,
    SUM(key_passes) / (SUM(minutes) / 90.0) AS key_passes_p90,
    SUM(pressures) / (SUM(minutes) / 90.0) AS pressures_p90,
    SUM(np_goals_assists) / (SUM(minutes) / 90.0) AS npga_p90,
    -- Raw totals for context
    SUM(np_goals) AS total_np_goals,
    SUM(assists) AS total_assists,
    SUM(np_goals_assists) AS total_npga
FROM player_match
GROUP BY player_id, player, team, primary_position
ORDER BY npga_p90 DESC;
