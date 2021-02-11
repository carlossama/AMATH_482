clear all; close all; clc
%% Playing the song

% figure(1)
[y, Fs] = audioread('GNR.m4a');
load piano.mat
%[y, Fs] = audioread('Floyd.m4a');
tr_gnr = length(y)/Fs; % record time in seconds
% plot((1:length(y))/Fs,y);
% xlabel('Time [sec]'); ylabel('Amplitude');
% title("Sweet Child O' Mine");
p8 = audioplayer(y,Fs); playblocking(p8);

%% Taking the Gabor Transform

% Parameters
t = (1:length(y))/Fs;
L = max(t);
n = length(t);
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
tau = 0:0.1:L;
alpha = 100;
ks = fftshift(k);

% Making sure my computer doesn't explode
easeUp = 10;
takeItEasy = zeros(length(y),1);
takeItEasy(1:easeUp:length(takeItEasy)) = 1;
Sgt_spec = zeros(length(find(takeItEasy)),length(tau));

% Performing the Gabor Transform
for j = 1:length(tau)
    Sgtf = zeros(length(yb),1);
    g = exp(-alpha*(t - tau(j)).^2); 
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgts = fftshift(abs(Sgt))';
    Sgts = Sgts(logical(takeItEasy));
    Sgt_spec(:,j) = Sgts; 
    
end

%% Plotting the Spectogram

figure
pcolor(tau,ks(logical(takeItEasy)),log(Sgt_spec+1))
shading interp
colormap(hot)
ylim([0, 8000])
colorbar
xlabel('time (t)'), ylabel('frequency (k)')

%% Reproducing the Song

t = (1:length(y))/Fs;
L = max(t);
n = length(t);
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
tau = 0:0.2:L;
alpha = 1000000;
trbBand = [1000, 3000];
song = zeros(0); 
freqs = zeros(length(tau),1);
kvals = zeros(length(tau),1);

for i = 1:length(tau)
    g = exp(-alpha*(t-tau(i)).^10);
    yf = g'.*y;
    yt = fft(yf);
    yt(k<trbBand(1)) = 0;
    yt(k>trbBand(2)) = 0;
    ind = ind2sub(n, find(yt == max(yt))); 
    kc = k(ind)/6.3;
    kvals(i) = kc;
    [val, idx] = min(abs(kc - piano_freq));
    freqs(i) = piano_freq(idx);       
    note = sin((2*pi)*freqs(i)*t);
    note = note(1:1:floor(length(y) / length(tau)));
    song = [song, note];
end
%% Generating the Score

i = 1;
songFreq = [];
notes = '';
while i <= length(freqs)
    freq_o = freqs(i);
    noteLen = 0;
    while (freqs(i) == freq_o) && (i <= length(freqs))
        noteLen = noteLen + 1;
        freq_o = freqs(i);
        i = i+1;
        if i > length(freqs)
            break  % sorry :(
        end 
    end
    if noteLen >= 2
       
        [val, noteIdx] = min(abs(freq_o - piano_freq));
        if ~isempty(notes)
            if ~strcmp(notes{end}, piano_notes{noteIdx})
                notes = strrep(strcat(notes, "_|_", piano_notes{noteIdx}),'_',' ');
            end
        else
             notes = strrep(strcat(notes, "_|_", piano_notes{noteIdx}),'_',' ');
        end                      
    end
end

    

%% Playing the Song

p8 = audioplayer(song*5,Fs); playblocking(p8); %filtered song


%% Isolating the bass


t = (1:length(y))/Fs;
L = max(t);
n = length(y);
k = (2*pi/L)*[0:n/2-1 -n/2:-1];
y = y(1:end-1);
yt = fft(y);

bassFilter = exp(-0.00002*((k - 650).^2))';

ytf = yt.*bassFilter;

yb = ifft(ytf);

%% Playing the Song

p8 = audioplayer(yb*10,Fs); playblocking(p8); 









