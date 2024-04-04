load primitive.mat

figure(1)
hold on

for i = 1:nbands
  plot((1:nkpts)/nkpts, eigval(:,i)-Efermi, 'ro-', 'MarkerSize', 1.0);
end


set(gca, 'YLim', [-3, 3]);
set(gca, 'XTick', []);

xlabel("$k$-path", 'Interpreter','latex');
ylabel("Energy (eV)", 'Interpreter',  'latex');

set(gca, 'XLim', [-Inf, Inf]);
set(gca, 'FontSize', 16);

box on

%%
load unfold.mat
[~,nkpt,nband,~] = size(sw);

energy = reshape(sw(1,:,:,1), nkpt, nband);
weight = reshape(sw(1,:,:,2), nkpt, nband);

xx = [1:nkpt]'/nkpt;
xx = kron(xx, ones(1,nband));


figure(24)
hold on

scatter(xx, energy-Efermi, weight*20, 'b', 'filled')

set(gca, 'YLim', [-3, 3]);
set(gca, 'XTick', []);

xlabel("$k$-path", 'Interpreter','latex');
ylabel("Energy (eV)", 'Interpreter',  'latex');

set(gca, 'XLim', [-Inf, Inf]);
set(gca, 'FontSize', 16);

box on



