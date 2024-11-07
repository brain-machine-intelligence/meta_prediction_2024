function [pressedKey]=getkeypress()
 WaitSecs(0.4);
 keyIsDown=0;
 while ~keyIsDown
    [KeyIsDown, pressedSecs, keyCode] = KbCheck(-1);
 end
 pressedKey = find(keyCode);
end
b