# #QuarentenaDados #Alura
   
Este repositório apresenta minha solução para o problema de regressão proposto pela [Alura #QuarentenaDados](https://www.alura.com.br/quarentenadados/aula05-regressao-machine-learning?utm_campaign=alura_quarentenadados_-_5_aula&utm_medium=email&utm_source=RD+Station).   
Resumidamente, dado as notas de um participante do ENEM em algumas matérias (NU_NOTA_CN, NU_NOTA_CH, NU_NOTA_MT, NU_NOTA_REDACAO), tente prever a nota de outra matéria (NU_NOTA_LC).   

---
## Solução
Minha solução foi baseada em encontrar um Regressor e otimizar a busca de parâmetros. Imagino que devesse ter realizado uma melhor análise sobre os dados, ou propor novas features - *Infelizmente, não consegui realizar antes da competição*.   
Nota-se que, observando os dados, não há ausência de valores e sua distribuição é massiva em torno da nota mediana.   

**Procedimentos**
- [00](dataset.py): Conjunto de dados.
- [01](01_exploring_regressors.py): Testar um conjunto de Regressores.
- [02](02_parameter_tuning.py): Realizar uma tuning de parâmetros na melhor solução.
- [03](03_refinement_test.py): Refinamento do modelo.
- [04](04_export_result.py): Exportando o resultado.

---
## Also look ~

-   [License MIT](LICENSE)
-   Create by Leonardo Mauro ~ [leomaurodesenv](https://github.com/leomaurodesenv/)
