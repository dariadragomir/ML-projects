class QwirkleScorer:
    def __init__(self, bonus_board):
        # matrice 16x16 care retine starea tablei sau None
        self.board_state = [[None for _ in range(16)] for _ in range(16)]
        
        # matrice de bonusuri 0, 1, 2 orietation2board
        self.bonus_board = bonus_board

    def update_and_score(self, new_pieces):
        # new_pieces: lista de tupluri (row_idx, col_idx, shape_code, color_code)
        # row_idx, col_idx sunt (0..15)
        turn_score = 0
        lines_formed = set() # set pentru a stoca liniile unice formate in aceasta tura
        
        for r, c, shape, color in new_pieces:
            if 0 <= r < 16 and 0 <= c < 16:
                self.board_state[r][c] = shape + color
            else:
                print(f"Eroare: Coordonate invalide {r},{c}")

        # calcul scor
        for r, c, shape, color in new_pieces:
            # bonusurile de pe tabla
            bonus_val = self.bonus_board[r][c]
            if bonus_val > 0:
                turn_score += bonus_val

            # detectarea liniilor
            # pe orizontala 0 si verticala 1
            directions = [(0, 1), (1, 0)] 
            
            for dr, dc in directions:
                line_coords = self._get_line_coordinates(r, c, dr, dc)
                
                # o linie valida are>= 2 piese
                if len(line_coords) >= 2:
                    # frozenset pentru a identifica linia in mod unic indiferent de ordine
                    line_signature = frozenset(line_coords)
                    
                    if line_signature not in lines_formed:
                        lines_formed.add(line_signature)
                        
                        # 1 punct per piesa inclusiv cele vechi din linie
                        points = len(line_coords)
                        turn_score += points
                        
                        if points == 6:
                            turn_score += 6
                            print("QWIRKLE! +6 puncte")

        return turn_score

    def _get_line_coordinates(self, r, c, dr, dc):
        #intr-o directie (dr, dc) si opusul ei pentru a gasi toata linia
        #returnez o lsita de coordonate (r, c) care fac parte din linie
        line = {(r, c)}
        
        # directia pozitiva deapta-jos
        curr_r, curr_c = r + dr, c + dc
        while 0 <= curr_r < 16 and 0 <= curr_c < 16:
            if self.board_state[curr_r][curr_c] is not None:
                line.add((curr_r, curr_c))
                curr_r += dr
                curr_c += dc
            else:
                break
        
        # directia negativa
        curr_r, curr_c = r - dr, c - dc
        while 0 <= curr_r < 16 and 0 <= curr_c < 16:
            if self.board_state[curr_r][curr_c] is not None:
                line.add((curr_r, curr_c))
                curr_r -= dr
                curr_c -= dc
            else:
                break
                
        return list(line)