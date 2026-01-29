from helpers import *
import os
from image_paths import * 
from game_logic import QwirkleScorer

os.makedirs(OUTPUT_FOLDER, exist_ok=True)  
os.makedirs(SAVE_TEMPLATES, exist_ok=True)

shape_templates = load_shape_templates(TEMPLATES_PIECES) 

all_game_images = import_images(PATH_IMAGES_TRAIN)

COUNTER = 1

for game_idx in range(NUM_GAMES): 
    print(f"\n=== Procesez JOCUL {game_idx + 1} ===")
    
    start_idx = game_idx * 21
    end_idx = start_idx + 21
    game_imgs = all_game_images[start_idx+1:end_idx]
    
    # move 00
    current_board_img = get_board(all_game_images[start_idx])
    orientation = get_orientation(current_board_img)
    bonus_board = orietation2board(orientation)
    scorer = QwirkleScorer(bonus_board)

    # pun pe tabla 24 piese initiale
    scan_initial_board(current_board_img, bonus_board, scorer, shape_templates, COUNTER)
    #print(scorer.board_state)
    
    last_processed_board = current_board_img

    # iterez prin mutari
    for move_idx, raw_img in enumerate(game_imgs, start=1):
        current_board_img = get_board(raw_img)
        
        output_filename = f"{game_idx + 1}_{move_idx:02d}.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        # detect piese noi adaugate
        diff_img = get_frame_diff(last_processed_board, current_board_img)
        changed_squares_indices = vote_for_square(diff_img)
        
        # lista pentru a stoca piesele acestei ture 
        turn_pieces_logic = [] # (row, col, shape, color)
        turn_pieces_text = []  # stringuri "10D 4G"
        
        print(f"  Mutarea {move_idx}: {len(changed_squares_indices)} piese noi.")
        
        for sq in changed_squares_indices:
            # sq[0] = rand (1-16), sq[1] = coloana (1-16)
            r, c = sq[0], sq[1]
            
            if(scorer.bonus_board[r-1][c-1] == -1):
                continue
            # extrag imaginea piesei
            piece_patch = get_square(current_board_img, r, c, p=17)
            # print(piece_patch)
            if piece_patch is None or piece_patch.size == 0:
                piece_patch = get_square(current_board_img, r, c, p=0)
                #padded_current_board_img = np.pad(current_board_img, 0, pad_with)
                #piece_patch = get_square(padded_current_board_img, r, c)
                # show_img(piece_patch)
                print(f"     Iau piesa fara padding la {r}, {c}")
                # exit(0) 
                # continue
            # get forma si culoare piesa

            shape_code, color_code, score = identify_piece(piece_patch, shape_templates, COUNTER)
            COUNTER += 1
            
            if shape_code and color_code: 
                turn_pieces_logic.append((r-1, c-1, shape_code, color_code))

                # coloana 1- 'A', 2- 'B'
                col_char = chr(ord('A') + c - 1)
                pos_str = f"{r}{col_char}"
                type_str = f"{shape_code}{color_code}"
                
                turn_pieces_text.append(f"{pos_str} {type_str}")
        
        # calcul scor
        move_score = 0
        if scorer:
            # procesez piesele in ordine
            turn_pieces_logic.sort() 
            move_score = scorer.update_and_score(turn_pieces_logic)
            print(f"    Scor calculat: {move_score}")
        
        
        # sortez liniile 
        turn_pieces_text.sort(key=lambda x: (int(x.split()[0][:-1]), x.split()[0][-1])) 
        
        with open(output_path, "w") as f:
            for line in turn_pieces_text:
                f.write(line + "\n")
            f.write(str(move_score))
        
        last_processed_board = current_board_img

print("\nProcesare completa. Fisierele sunt in folderul: 352_Dragomir_Daria_Nicoleta/output")
