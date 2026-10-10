# 模擬119通報30件の音声と文字起こし

雑音付き音声をCER評価に使用した。Gemini元音声と各台本も保存している。

| 番号 | 内容 | 長さ | 雑音付き | Gemini元音声 | 文字起こし |
|---:|---|---:|---|---|---|
| 01 | 胸痛の救急通報 | 21.7秒 | [再生](audio/telephone/mock_01_chest_pain.wav) | [再生](audio/gemini_clean/mock_01_chest_pain.wav) | [読む](transcripts/mock_01_chest_pain.txt) |
| 02 | 台所の火災通報 | 20.3秒 | [再生](audio/telephone/mock_02_kitchen_fire.wav) | [再生](audio/gemini_clean/mock_02_kitchen_fire.wav) | [読む](transcripts/mock_02_kitchen_fire.txt) |
| 03 | 交通事故の救急通報 | 20.6秒 | [再生](audio/telephone/mock_03_bicycle_crash.wav) | [再生](audio/gemini_clean/mock_03_bicycle_crash.wav) | [読む](transcripts/mock_03_bicycle_crash.txt) |
| 04 | 階段での転倒の救急通報 | 18.6秒 | [再生](audio/telephone/mock_04_stair_fall.wav) | [再生](audio/gemini_clean/mock_04_stair_fall.wav) | [読む](transcripts/mock_04_stair_fall.txt) |
| 05 | 建物の煙の火災通報 | 21.5秒 | [再生](audio/telephone/mock_05_building_smoke.wav) | [再生](audio/gemini_clean/mock_05_building_smoke.wav) | [読む](transcripts/mock_05_building_smoke.txt) |
| 06 | 子どものけいれんの救急通報 | 17.9秒 | [再生](audio/telephone/mock_06_child_seizure.wav) | [再生](audio/gemini_clean/mock_06_child_seizure.wav) | [読む](transcripts/mock_06_child_seizure.txt) |
| 07 | 川への転落の救助通報 | 17.2秒 | [再生](audio/telephone/mock_07_river_fall.wav) | [再生](audio/gemini_clean/mock_07_river_fall.wav) | [読む](transcripts/mock_07_river_fall.txt) |
| 08 | 工場での負傷の救急通報 | 22.2秒 | [再生](audio/telephone/mock_08_factory_injury.wav) | [再生](audio/gemini_clean/mock_08_factory_injury.wav) | [読む](transcripts/mock_08_factory_injury.txt) |
| 09 | 公園での熱中症疑いの救急通報 | 20.8秒 | [再生](audio/telephone/mock_09_heat_illness.wav) | [再生](audio/gemini_clean/mock_09_heat_illness.wav) | [読む](transcripts/mock_09_heat_illness.txt) |
| 10 | ガス臭の火災予防通報 | 19.3秒 | [再生](audio/telephone/mock_10_gas_smell.wav) | [再生](audio/gemini_clean/mock_10_gas_smell.wav) | [読む](transcripts/mock_10_gas_smell.txt) |
| 11 | 飲食店での意識消失 | 19.2秒 | [再生](audio/telephone/mock_11_restaurant_collapse.wav) | [再生](audio/gemini_clean/mock_11_restaurant_collapse.wav) | [読む](transcripts/mock_11_restaurant_collapse.txt) |
| 12 | ベランダからの出火 | 19.8秒 | [再生](audio/telephone/mock_12_balcony_fire.wav) | [再生](audio/gemini_clean/mock_12_balcony_fire.wav) | [読む](transcripts/mock_12_balcony_fire.txt) |
| 13 | プールでの救急通報 | 18.8秒 | [再生](audio/telephone/mock_13_pool_rescue.wav) | [再生](audio/gemini_clean/mock_13_pool_rescue.wav) | [読む](transcripts/mock_13_pool_rescue.txt) |
| 14 | バイク事故 | 19.3秒 | [再生](audio/telephone/mock_14_motorcycle_crash.wav) | [再生](audio/gemini_clean/mock_14_motorcycle_crash.wav) | [読む](transcripts/mock_14_motorcycle_crash.txt) |
| 15 | 食後の呼吸困難 | 20.2秒 | [再生](audio/telephone/mock_15_food_allergy.wav) | [再生](audio/gemini_clean/mock_15_food_allergy.wav) | [読む](transcripts/mock_15_food_allergy.txt) |
| 16 | 河川敷の草火災 | 20.6秒 | [再生](audio/telephone/mock_16_grass_fire.wav) | [再生](audio/gemini_clean/mock_16_grass_fire.wav) | [読む](transcripts/mock_16_grass_fire.txt) |
| 17 | エレベーター閉じ込め | 18.5秒 | [再生](audio/telephone/mock_17_elevator_trapped.wav) | [再生](audio/gemini_clean/mock_17_elevator_trapped.wav) | [読む](transcripts/mock_17_elevator_trapped.txt) |
| 18 | 出産間近の救急通報 | 19.2秒 | [再生](audio/telephone/mock_18_labor.wav) | [再生](audio/gemini_clean/mock_18_labor.wav) | [読む](transcripts/mock_18_labor.txt) |
| 19 | 学校の校舎から煙 | 20.1秒 | [再生](audio/telephone/mock_19_school_smoke.wav) | [再生](audio/gemini_clean/mock_19_school_smoke.wav) | [読む](transcripts/mock_19_school_smoke.txt) |
| 20 | 駅のホームでの急病 | 18.2秒 | [再生](audio/telephone/mock_20_station_collapse.wav) | [再生](audio/gemini_clean/mock_20_station_collapse.wav) | [読む](transcripts/mock_20_station_collapse.txt) |
| 21 | 車庫内の火災 | 17.2秒 | [再生](audio/telephone/mock_21_garage_fire.wav) | [再生](audio/gemini_clean/mock_21_garage_fire.wav) | [読む](transcripts/mock_21_garage_fire.txt) |
| 22 | 登山道での負傷 | 21.4秒 | [再生](audio/telephone/mock_22_hiking_injury.wav) | [再生](audio/gemini_clean/mock_22_hiking_injury.wav) | [読む](transcripts/mock_22_hiking_injury.txt) |
| 23 | 食べ物を喉に詰まらせた通報 | 18.4秒 | [再生](audio/telephone/mock_23_choking.wav) | [再生](audio/gemini_clean/mock_23_choking.wav) | [読む](transcripts/mock_23_choking.txt) |
| 24 | 暖房器具使用中の体調不良 | 20.0秒 | [再生](audio/telephone/mock_24_heater_fumes.wav) | [再生](audio/gemini_clean/mock_24_heater_fumes.wav) | [読む](transcripts/mock_24_heater_fumes.txt) |
| 25 | 遊具からの転落 | 17.8秒 | [再生](audio/telephone/mock_25_playground_fall.wav) | [再生](audio/gemini_clean/mock_25_playground_fall.wav) | [読む](transcripts/mock_25_playground_fall.txt) |
| 26 | 道路脇の土砂崩れ | 21.6秒 | [再生](audio/telephone/mock_26_slope_collapse.wav) | [再生](audio/gemini_clean/mock_26_slope_collapse.wav) | [読む](transcripts/mock_26_slope_collapse.txt) |
| 27 | 倉庫での刺激臭 | 19.0秒 | [再生](audio/telephone/mock_27_warehouse_odor.wav) | [再生](audio/gemini_clean/mock_27_warehouse_odor.wav) | [読む](transcripts/mock_27_warehouse_odor.txt) |
| 28 | 急な腹痛の救急通報 | 19.6秒 | [再生](audio/telephone/mock_28_abdominal_pain.wav) | [再生](audio/gemini_clean/mock_28_abdominal_pain.wav) | [読む](transcripts/mock_28_abdominal_pain.txt) |
| 29 | 浴室での意識低下 | 17.8秒 | [再生](audio/telephone/mock_29_bathroom_collapse.wav) | [再生](audio/gemini_clean/mock_29_bathroom_collapse.wav) | [読む](transcripts/mock_29_bathroom_collapse.txt) |
| 30 | 路線バスからの煙 | 19.5秒 | [再生](audio/telephone/mock_30_bus_smoke.wav) | [再生](audio/gemini_clean/mock_30_bus_smoke.wav) | [読む](transcripts/mock_30_bus_smoke.txt) |
